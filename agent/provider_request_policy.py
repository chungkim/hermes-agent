"""Generic provider request-policy seam and safety-critical enforcement primitives.

Model/provider plugins implement :class:`ProviderRequestPolicyPlugin` and register
through this module. The core owns validation, bounded admission, deadline data,
and terminal error markers; route identity and provider-specific configuration stay
in the plugin.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = 8
DEFAULT_MAX_QUEUED = 4
MIN_QUEUE_TIMEOUT = 5.0
MAX_QUEUE_TIMEOUT = 180.0
MIN_FIRST_BYTE_TIMEOUT = 90.0
MAX_FIRST_BYTE_TIMEOUT = 180.0
MIN_IDLE_TIMEOUT = 90.0
MAX_IDLE_TIMEOUT = 180.0
MIN_TOTAL_TIMEOUT = 300.0
MAX_TOTAL_TIMEOUT = 600.0
MAX_CONFIG_MAX_CONCURRENT = DEFAULT_MAX_CONCURRENT
MAX_CONFIG_MAX_QUEUED = DEFAULT_MAX_QUEUED


class ProviderAdmissionRejected(RuntimeError):
    """A bounded provider admission queue rejected a request before transport."""

    admission_rejected = True
    retryable = False


class ProviderRequestDeadlineExceeded(TimeoutError):
    """A selected provider request budget expired; replay is unsafe by default."""

    provider_request_deadline = True
    retryable = False


@dataclass(frozen=True)
class ProviderRequestPolicy:
    """Validated route policy and bounded derived deadlines, in seconds."""

    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    max_queued: int = DEFAULT_MAX_QUEUED
    queue_timeout: float = MIN_QUEUE_TIMEOUT
    first_byte_timeout: float = MIN_FIRST_BYTE_TIMEOUT
    idle_timeout: float = MIN_IDLE_TIMEOUT
    total_timeout: float = MIN_TOTAL_TIMEOUT

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ProviderRequestPolicy:
        """Validate explicit values without silently changing global settings."""
        if not isinstance(values, dict):
            raise ValueError("provider request policy must be a mapping")

        def positive(name: str, default: float, *, minimum: float, maximum: float) -> float:
            value = values.get(name, default)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a positive number")
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a positive number") from None
            if not math.isfinite(value) or value < minimum or value > maximum:
                raise ValueError(f"{name} must be between {minimum:g} and {maximum:g} seconds")
            return value

        def bounded_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
            value = values.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
            return value

        return cls(
            max_concurrent=bounded_int("max_concurrent_requests", DEFAULT_MAX_CONCURRENT, minimum=1, maximum=MAX_CONFIG_MAX_CONCURRENT),
            max_queued=bounded_int("max_queued_requests", DEFAULT_MAX_QUEUED, minimum=0, maximum=MAX_CONFIG_MAX_QUEUED),
            queue_timeout=positive("admission_timeout_seconds", MIN_QUEUE_TIMEOUT, minimum=MIN_QUEUE_TIMEOUT, maximum=MAX_QUEUE_TIMEOUT),
            first_byte_timeout=positive("first_byte_timeout_seconds", MIN_FIRST_BYTE_TIMEOUT, minimum=MIN_FIRST_BYTE_TIMEOUT, maximum=MAX_FIRST_BYTE_TIMEOUT),
            idle_timeout=positive("stream_read_timeout_seconds", MIN_IDLE_TIMEOUT, minimum=MIN_IDLE_TIMEOUT, maximum=MAX_IDLE_TIMEOUT),
            total_timeout=positive("request_timeout_seconds", MIN_TOTAL_TIMEOUT, minimum=MIN_TOTAL_TIMEOUT, maximum=MAX_TOTAL_TIMEOUT),
        )

    def with_derived_budgets(
        self,
        *,
        context_tokens: int = 0,
        admission_position: int = 0,
        observed_queue_wait: float = 0.0,
        active_requests: int = 0,
    ) -> ProviderRequestPolicy:
        """Derive bounded queue, first-byte, idle, and total deadlines."""
        del active_requests
        tokens = max(0, int(context_tokens or 0))
        context_k = min(tokens / 1000.0, 300.0)
        pressure = max(0.0, float(observed_queue_wait or 0.0))
        pressure_bonus = min(45.0, pressure * 0.5)
        queue = min(MAX_QUEUE_TIMEOUT, max(MIN_QUEUE_TIMEOUT, max(30.0, self.queue_timeout) + admission_position * 15.0 + pressure_bonus))
        first = min(MAX_FIRST_BYTE_TIMEOUT, max(MIN_FIRST_BYTE_TIMEOUT, self.first_byte_timeout + context_k / 4.0))
        idle = min(MAX_IDLE_TIMEOUT, max(MIN_IDLE_TIMEOUT, self.idle_timeout + context_k / 8.0))
        total = min(MAX_TOTAL_TIMEOUT, max(MIN_TOTAL_TIMEOUT, self.total_timeout + context_k))
        return ProviderRequestPolicy(
            max_concurrent=self.max_concurrent,
            max_queued=self.max_queued,
            queue_timeout=queue,
            first_byte_timeout=first,
            idle_timeout=idle,
            total_timeout=total,
        )


class ProviderRequestPolicyPlugin(ABC):
    """Provider-neutral plugin contract for selecting a route policy."""

    name: str

    @abstractmethod
    def policy_for_route(
        self, *, provider: str | None, model: str | None, base_url: str | None
    ) -> ProviderRequestPolicy | None:
        """Return a policy only for routes owned by this plugin."""

    @staticmethod
    def policy_from_mapping(values: dict[str, Any]) -> ProviderRequestPolicy:
        return ProviderRequestPolicy.from_mapping(values)

    def admission_key(self, *, provider: str | None, model: str | None, base_url: str | None) -> tuple[str, ...]:
        """Stable route key; plugins may include additional route dimensions."""
        return (
            str(provider or "").strip().lower(),
            str(model or "").strip().lower(),
            str(base_url or "").strip().lower(),
        )


_POLICY_SCOPE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "provider_request_policy_scope", default=None
)
_POLICY_PLUGINS: dict[str | None, dict[str, ProviderRequestPolicyPlugin]] = {None: {}}
_POLICY_LOCK = threading.RLock()
_POLICY_DISCOVERY_DONE: set[str | None] = set()


@contextlib.contextmanager
def request_policy_scope(scope: str | None) -> Iterator[None]:
    """Bind plugin registrations to a provider-discovery scope."""
    token = _POLICY_SCOPE.set(scope)
    try:
        yield
    finally:
        _POLICY_SCOPE.reset(token)


def register_request_policy_plugin(plugin: ProviderRequestPolicyPlugin, *, scope: str | None = None) -> None:
    """Register one plugin in the current discovery scope."""
    if not isinstance(plugin, ProviderRequestPolicyPlugin):
        raise TypeError("request policy plugin must inherit ProviderRequestPolicyPlugin")
    if not isinstance(plugin.name, str) or not plugin.name.strip():
        raise ValueError("request policy plugin .name must be a non-empty string")
    target = _POLICY_SCOPE.get() if scope is None else scope
    with _POLICY_LOCK:
        _POLICY_PLUGINS.setdefault(target, {})[plugin.name.strip().lower()] = plugin


def list_request_policy_plugins() -> list[ProviderRequestPolicyPlugin]:
    """Return global plus active-home plugins, with scoped overrides last."""
    try:
        from hermes_constants import get_hermes_home, hermes_home_key
        active_scope = hermes_home_key(get_hermes_home())
    except Exception:
        active_scope = None
    with _POLICY_LOCK:
        merged = dict(_POLICY_PLUGINS.get(None, {}))
        if active_scope is not None:
            merged.update(_POLICY_PLUGINS.get(active_scope, {}))
        return list(merged.values())


def _ensure_policy_plugins_discovered() -> None:
    try:
        from hermes_constants import get_hermes_home, hermes_home_key
        scope = hermes_home_key(get_hermes_home())
    except Exception:
        scope = None
    with _POLICY_LOCK:
        if scope in _POLICY_DISCOVERY_DONE:
            return
        _POLICY_DISCOVERY_DONE.add(scope)
    try:
        from providers import list_providers
        list_providers()
    except Exception:
        logger.debug("request policy plugin discovery failed", exc_info=True)


def automatic_policy_for_route(
    provider: str | None, model: str | None, base_url: str | None
) -> ProviderRequestPolicy | None:
    """Resolve the first plugin-owned policy; absent means legacy core defaults."""
    _ensure_policy_plugins_discovered()
    for plugin in list_request_policy_plugins():
        try:
            policy = plugin.policy_for_route(provider=provider, model=model, base_url=base_url)
        except Exception:
            logger.warning("request policy plugin %s failed closed", plugin.name, exc_info=True)
            continue
        if policy is not None:
            return policy
    return None


class AdmissionLease:
    def __init__(self, owner: ProviderAdmission, *, queue_position: int, queue_wait_seconds: float):
        self._owner = owner
        self.queue_position = queue_position
        self.queue_wait_seconds = queue_wait_seconds
        self.released = False
        self._release_lock = threading.Lock()

    def release(self, reason: str = "complete") -> None:
        with self._release_lock:
            if self.released:
                return
            self.released = True
        self._owner._release(reason)

    def __enter__(self) -> AdmissionLease:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release("exception" if exc_type else "complete")


@dataclass
class _AdmissionWaiter:
    ticket: int


class ProviderAdmission:
    """A bounded FIFO admission gate shared by one resolved route."""

    def __init__(self, *, max_concurrent: int, max_queued: int):
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be an integer >= 1")
        if isinstance(max_queued, bool) or not isinstance(max_queued, int) or max_queued < 0:
            raise ValueError("max_queued must be an integer >= 0")
        self.max_concurrent = max_concurrent
        self.max_queued = max_queued
        self._condition = threading.Condition(threading.RLock())
        self._active = 0
        self._waiters: deque[_AdmissionWaiter] = deque()
        self._next_ticket = 0
        self._peak_active = 0
        self._last_queue_wait = 0.0
        self._rejections = 0

    def acquire(self, *, timeout: float | None = None) -> AdmissionLease:
        start = time.monotonic()
        with self._condition:
            if self._active < self.max_concurrent and not self._waiters:
                self._active += 1
                self._peak_active = max(self._peak_active, self._active)
                return AdmissionLease(self, queue_position=0, queue_wait_seconds=0.0)
            if len(self._waiters) >= self.max_queued:
                self._rejections += 1
                raise ProviderAdmissionRejected(
                    f"provider admission capacity reached (active={self._active}, max_concurrent={self.max_concurrent}, max_queued={self.max_queued})"
                )
            if timeout is not None and timeout <= 0:
                self._rejections += 1
                raise ProviderAdmissionRejected("provider admission queue is full for an immediate request")
            self._next_ticket += 1
            waiter = _AdmissionWaiter(self._next_ticket)
            self._waiters.append(waiter)
            position = len(self._waiters)
            try:
                deadline = None if timeout is None else start + timeout
                while True:
                    if self._active < self.max_concurrent and self._waiters and self._waiters[0] is waiter:
                        self._waiters.popleft()
                        self._active += 1
                        self._peak_active = max(self._peak_active, self._active)
                        waited = time.monotonic() - start
                        self._last_queue_wait = waited
                        return AdmissionLease(self, queue_position=position, queue_wait_seconds=waited)
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        self._rejections += 1
                        raise ProviderAdmissionRejected(
                            f"provider admission queue wait exceeded {timeout:.1f}s (position={position})"
                        )
                    self._condition.wait(remaining)
            finally:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(waiter)
                self._condition.notify_all()

    def _release(self, reason: str) -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify_all()
            logger.info(
                "Provider admission slot released (reason=%s active=%d waiting=%d max=%d; server-side cancellation is not implied)",
                reason, self._active, len(self._waiters), self.max_concurrent,
            )

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {"active": self._active, "waiting": len(self._waiters), "max_concurrent": self.max_concurrent, "max_queued": self.max_queued}

    def telemetry(self) -> dict[str, float | int]:
        with self._condition:
            return {**self.snapshot(), "peak_active": self._peak_active, "last_queue_wait_seconds": round(self._last_queue_wait, 3), "rejections": self._rejections}

    @contextlib.contextmanager
    def lease(self, *, timeout: float | None = None):
        lease = self.acquire(timeout=timeout)
        try:
            yield lease
        except BaseException:
            lease.release("exception")
            raise
        else:
            lease.release("complete")


_ADMISSIONS: dict[tuple[str, ...], ProviderAdmission] = {}
_ADMISSIONS_LOCK = threading.Lock()


def admission_for_route(provider: str | None, model: str | None, base_url: str | None) -> ProviderAdmission | None:
    """Return a gate for a plugin-owned route, or None for compatibility defaults."""
    _ensure_policy_plugins_discovered()
    for plugin in list_request_policy_plugins():
        policy = plugin.policy_for_route(provider=provider, model=model, base_url=base_url)
        if policy is None:
            continue
        key = plugin.admission_key(provider=provider, model=model, base_url=base_url)
        with _ADMISSIONS_LOCK:
            return _ADMISSIONS.setdefault(
                key, ProviderAdmission(max_concurrent=policy.max_concurrent, max_queued=policy.max_queued)
            )
    return None


def reset_request_policy_for_tests() -> None:
    with _POLICY_LOCK:
        _POLICY_PLUGINS.clear()
        _POLICY_PLUGINS[None] = {}
        _POLICY_DISCOVERY_DONE.clear()
    with _ADMISSIONS_LOCK:
        _ADMISSIONS.clear()


__all__ = [
    "AdmissionLease", "ProviderAdmission", "ProviderAdmissionRejected", "ProviderRequestDeadlineExceeded",
    "ProviderRequestPolicy", "ProviderRequestPolicyPlugin", "admission_for_route",
    "automatic_policy_for_route", "list_request_policy_plugins", "register_request_policy_plugin",
    "request_policy_scope", "reset_request_policy_for_tests",
]
