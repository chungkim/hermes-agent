"""Display and heartbeat phase of the request-local streaming monitor."""

import time
from types import SimpleNamespace

from agent import chat_completion_wait_notice as wn
from agent.model_metadata import is_local_endpoint


class StreamingWaitMonitor:
    def _stream_liveness(self) -> dict:
        """Return the current attempt's monotonic liveness state."""
        getter = getattr(self, "_current_attempt_liveness", None)
        if callable(getter):
            state = getter()
            if isinstance(state, dict):
                return state
        # Older test doubles and integrations may not provide attempt-local state.
        # Preserve their wall-clock chunk marker while adapting it to the monitor's
        # monotonic liveness contract; returning only a monotonic timestamp would
        # make resumed chunks invisible to the wait-notice clearer.
        now_mono = time.monotonic()
        now_wall = time.time()
        last_wall = now_wall
        try:
            last_wall = float(self.last_chunk_time["t"])
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        elapsed = max(0.0, now_wall - last_wall)
        return {
            "attempt_id": 0,
            "last_progress_at": now_mono - elapsed - 1e-6,
            "last_progress_wall": last_wall,
            "first_chunk_at": None,
        }

    def _poll_local_load_notice(self, now: float) -> bool:
        """Managed local server: surface a cold model's weight-load progress
        instead of the 60s neutral "waiting on <model>" notice. Polled ~1s only while no
        REAL chunk arrived for 2s+ (never during healthy token flow); in-memory,
        no network. True while loading = heartbeat liveness, skip the rest of
        this iteration (the stale detector's local floor dwarfs any load)."""
        from agent.chat_completion_helpers import _managed_local_load_notice

        m = self._mon
        liveness = self._stream_liveness()
        legacy_clock = liveness.get("attempt_id") == 0
        poll_now = time.time() if legacy_clock else now
        last_progress = liveness["last_progress_wall"] if legacy_clock else liveness["last_progress_at"]
        if poll_now - last_progress < 2.0 or poll_now - m.last_load_poll < 1.0:
            return False
        m.last_load_poll = poll_now
        _load_notice = _managed_local_load_notice(self.agent, self.api_kwargs)
        if _load_notice is not None:
            m.wait_notice_started_ts = None  # The local loader now owns the display.
            m.wait_notice.reset()
            self.agent._emit_wait_notice(_load_notice)
            self.agent._touch_activity("local model loading")
            # ``last_heartbeat`` is compared with wall-clock samples in the monitor
            # loop; initialized attempts pass a monotonic ``now`` for polling.
            m.load_notice_shown, m.load_notice_misses, m.last_heartbeat = True, 0, time.time()  # loading IS liveness
            return True
        if m.load_notice_shown:
            # One missed sample is routine (probe timeout under load); clearing on it strobed the line.
            m.load_notice_misses += 1
            if m.load_notice_misses >= 3:
                m.load_notice_shown, m.load_notice_misses = False, 0
                self.agent._emit_wait_notice("")
        return False

    def _heartbeat(self, waiting_secs: int) -> None:
        """Gateway inactivity heartbeat: the start-to-first-chunk gap (thinking,
        local prefill) can exceed the gateway timeout."""
        if waiting_secs >= 60.0:
            # No chunks for 60s+: say WHAT the wait is and WHEN recovery kicks in —
            # once per silence, not every heartbeat (#92550).
            stale = self._stream_stale_timeout
            watchdog = ("stream stale", stale - waiting_secs) if stale is not None and stale != float("inf") else None
            diag = getattr(getattr(self, "clients", None), "diag", None)
            phase = "post_chunk" if isinstance(diag, dict) and diag.get("first_chunk_at") else "first_chunk"
            if not self._mon.wait_notice.should_emit(phase, watchdog):
                self.agent._touch_activity(f"waiting for stream response ({waiting_secs}s, {phase})")
                return
            self._mon.wait_notice_started_ts = self._mon.last_heartbeat
            self.agent._emit_wait_notice(wn.wait_notice_text(
                self.api_kwargs.get('model', 'the provider'), waiting_secs, phase, watchdog))
        else:
            # Chunks are flowing — keep the tracker fresh, leave the display alone.
            self.agent._touch_activity(f"waiting for stream response ({waiting_secs}s, no chunks yet)")

    def _monitor_loop(self) -> None:
        _HEARTBEAT_INTERVAL = 30.0  # seconds between gateway activity touches
        self._mon = SimpleNamespace(
            last_heartbeat=time.time(), last_load_poll=0.0,
            load_notice_shown=False, load_notice_misses=0, wait_notice_started_ts=None,
            wait_notice=wn.WaitNoticeState(),
        )
        _is_local_base = bool(self.agent.base_url) and is_local_endpoint(self.agent.base_url)
        while not self._call_done.is_set():
            self._call_done.wait(timeout=0.3)
            _hb_now = time.time()
            _mono_now = time.monotonic()
            _liveness = self._stream_liveness()
            _legacy_liveness = _liveness.get("attempt_id") == 0
            if _is_local_base and self._poll_local_load_notice(_mono_now):
                continue
            # Reasoning callbacks do not clear the classic CLI spinner. The empty
            # protocol payload resets status without adding synthetic reasoning.
            if (self._mon.wait_notice_started_ts is not None
                    and _liveness.get("last_progress_wall", 0.0) > self._mon.wait_notice_started_ts):
                self.agent._emit_wait_notice("")
                self._mon.wait_notice_started_ts = None
                self._mon.wait_notice.reset()
            if _hb_now - self._mon.last_heartbeat >= _HEARTBEAT_INTERVAL:
                self._mon.last_heartbeat = _hb_now
                _waiting_elapsed = (
                    _hb_now - _liveness["last_progress_wall"]
                    if _legacy_liveness else _mono_now - _liveness["last_progress_at"]
                )
                self._heartbeat(int(_waiting_elapsed))
            _started = getattr(self, "_stream_started_at", None)
            _total_timeout = getattr(self, "_stream_total_timeout", None)
            _kill_total = getattr(self, "_kill_total_stream", None)
            if _started is not None and _total_timeout is not None and callable(_kill_total):
                _elapsed = time.monotonic() - _started
                if _elapsed >= _total_timeout:
                    _kill_total(_elapsed)
                    continue
            _stale_elapsed = (
                _hb_now - _liveness["last_progress_wall"]
                if _legacy_liveness else _mono_now - _liveness["last_progress_at"]
            )
            _first_chunk_seen = bool(_liveness.get("first_chunk_at"))
            _stale_limit = getattr(self, "_stream_stale_timeout", float("inf"))
            if not _first_chunk_seen:
                _stale_limit = getattr(self, "_stream_first_byte_timeout", None) or _stale_limit
            if _stale_elapsed > _stale_limit:
                self._mon.wait_notice_started_ts = None  # Reconnect status has its own owner.
                self._mon.wait_notice.reset()
                self._kill_stale_stream(_stale_elapsed)
            if self.agent._interrupt_requested:
                self._abort_for_interrupt(_stale_elapsed)
                return

