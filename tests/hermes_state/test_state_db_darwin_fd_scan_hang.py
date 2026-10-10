"""A kernel-blocked vnode must not wedge Hermes startup (#113187).

``_iter_darwin_fd_targets`` calls ``proc_pidfdinfo`` for every descriptor of
every process. If any process holds a descriptor whose vnode lookup blocks in
the kernel (e.g. uninterruptible I/O on a dead network share), that call never
returns -- and it used to run synchronously on the SessionDB connect path, so
one stuck unrelated process froze every Hermes start indefinitely.

``_iter_darwin_sidecar_holders`` must therefore bound the whole pass with a
wall-clock deadline (a per-iteration check cannot fire while blocked inside a
single call) and fail open with a warning. libproc is faked throughout, so
these run on every platform.
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import threading
import time

import hermes_state_dbfile as dbfile
from hermes_state_dbfile import (
    _DARWIN_FD_INO_OFFSET,
    _DARWIN_FD_PATH_OFFSET,
    _DARWIN_FD_DEV_OFFSET,
    _DARWIN_FD_RECORD_SIZE,
    _iter_darwin_sidecar_holders,
)

# Far below the 5s production budget so the suite stays fast; far above zero
# so scheduling jitter cannot flake it.
BUDGET = 2.0
# Guard so a regression (scan blocks again) fails instead of hanging the suite.
HANG_GUARD = 30.0


class _HungLibproc:
    """One pid, one fd, whose vnode lookup never returns (state-U process)."""

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        payload = struct.pack("<iI", 7, 1)
        ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        threading.Event().wait()
        return 0  # unreachable


class _HealthyLibproc:
    """One pid, one vnode fd naming ``target`` with the given identity."""

    def __init__(self, target, identity):
        self._target = target
        self._identity = identity

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        payload = struct.pack("<iI", 7, 1)
        ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        blob = bytearray(_DARWIN_FD_RECORD_SIZE)
        struct.pack_into("<I", blob, _DARWIN_FD_DEV_OFFSET, self._identity[0])
        struct.pack_into("<Q", blob, _DARWIN_FD_INO_OFFSET, self._identity[1])
        encoded = self._target.encode("utf-8")
        blob[_DARWIN_FD_PATH_OFFSET:_DARWIN_FD_PATH_OFFSET + len(encoded)] = encoded
        ctypes.memmove(record, bytes(blob), len(blob))
        return 1


def _run_guarded(func, *args):
    """Run ``func`` so a hang fails the test instead of hanging the suite."""
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(func(*args)), daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(HANG_GUARD)
    return worker, time.monotonic() - started, outcome


def test_hung_vnode_scan_fails_open_within_budget(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: _HungLibproc())
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    if hasattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS"):
        monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS", BUDGET)
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        worker, elapsed, outcome = _run_guarded(
            _iter_darwin_sidecar_holders, tmp_path / "state.db"
        )
    assert not worker.is_alive(), "scan blocked past the hang guard: no deadline"
    assert elapsed < HANG_GUARD
    assert outcome == [[]], "a timed-out scan must fail open (no holders, no refusal)"
    assert any(
        "timed out" in record.getMessage() for record in caplog.records
    ), "deadline expiry must log a visible warning"


def test_healthy_scan_still_reports_holder(monkeypatch, tmp_path, caplog):
    """The timeout wrapper must not swallow a fast, successful scan."""
    watched = os.path.realpath(os.path.join(os.fspath(tmp_path), "state.db-wal"))
    # The watched path names nothing: stat fails, so the fd's identity is
    # judged a retired generation and the holder is reported.
    lib = _HealthyLibproc(watched, (0xDEAD, 0xBEEF))
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    if hasattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS"):
        monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS", BUDGET)
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        worker, elapsed, outcome = _run_guarded(
            _iter_darwin_sidecar_holders, tmp_path / "state.db"
        )
    assert not worker.is_alive()
    assert elapsed < BUDGET, "a fast scan must not wait out the budget"
    assert outcome == [[(4242, watched)]]
    assert not any(
        "timed out" in record.getMessage() for record in caplog.records
    ), "a successful scan must not log a timeout warning"


# ── Scan cost (Docker Desktop VM): one process can hold ~200k descriptors ─────────────────────
#
# The Docker Desktop VM keeps a host fd per virtiofs inode, so a single pid can list hundreds of
# thousands of descriptors.  The pass must stay linear in that count (reading ``listing.raw`` per
# fd copied the whole listing each time: quadratic, minutes per pass), must not ask
# ``proc_pidfdinfo`` about non-vnode descriptors, and an abandoned slow pass must stop instead of
# pinning a core for the life of the process.

class _ManyFdLibproc:
    """One pid listing *count* descriptors; every ``skip_every``-th one is a socket, not a vnode."""

    def __init__(self, count, skip_every=0, delay=0.0):
        table = bytearray()
        for fd in range(count):
            fdtype = 2 if skip_every and fd % skip_every == 0 else 1
            table += struct.pack("<iI", fd, fdtype)
        self._table = bytes(table)
        self._delay = delay
        self.asked = []

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        if size <= len(self._table):
            ctypes.memmove(buf, self._table, size)
            return size  # buffer full: the caller must grow it
        ctypes.memmove(buf, self._table, len(self._table))
        return len(self._table)

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        self.asked.append(fd)
        if self._delay:
            time.sleep(self._delay)
        return 0  # not a describable vnode: skipped


def test_scan_is_linear_in_descriptor_count(monkeypatch, tmp_path, caplog):
    lib = _ManyFdLibproc(150_000)
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS", HANG_GUARD)
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        worker, elapsed, outcome = _run_guarded(
            _iter_darwin_sidecar_holders, tmp_path / "state.db"
        )
    assert not worker.is_alive()
    assert outcome == [[]]
    assert len(lib.asked) == 150_000, "every vnode descriptor must still be inspected"
    assert elapsed < BUDGET * 2, f"150k-fd pass took {elapsed:.1f}s: per-fd cost is not constant"


def test_non_vnode_descriptors_never_reach_pidfdinfo(monkeypatch, tmp_path):
    lib = _ManyFdLibproc(100, skip_every=3)
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    assert _iter_darwin_sidecar_holders(tmp_path / "state.db") == []
    assert lib.asked == [fd for fd in range(100) if fd % 3 != 0]


def test_abandoned_slow_scan_stops_consuming(monkeypatch, tmp_path):
    lib = _ManyFdLibproc(100_000, delay=0.001)  # ~100s if it ran to the end
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS", 0.3)
    assert _iter_darwin_sidecar_holders(tmp_path / "state.db") == []
    time.sleep(0.2)  # the in-flight call finishes, then the pass sees the stop flag
    settled = len(lib.asked)
    time.sleep(0.5)
    assert len(lib.asked) == settled, "a timed-out scan kept running after it was abandoned"
    assert settled < 100_000
