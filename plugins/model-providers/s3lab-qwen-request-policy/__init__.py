"""Route policy plugin for the exact long-context endpoint/model pair.

This directory is independently installable under ``$HERMES_HOME/plugins/model-providers``.
It owns route matching, configuration lookup, and provider-specific compatibility
helpers; the generic core only sees ProviderRequestPolicyPlugin.
"""

from __future__ import annotations

import contextlib
import math
import re
import threading
from dataclasses import dataclass
from urllib.parse import urlparse

from agent.provider_request_policy import (
    DEFAULT_MAX_CONCURRENT,
    ProviderAdmission,
    ProviderAdmissionRejected,
    ProviderRequestPolicy,
    ProviderRequestPolicyPlugin,
    register_request_policy_plugin,
)

_POLICY_KEYS = frozenset(
    {
        "max_concurrent_requests",
        "max_queued_requests",
        "admission_timeout_seconds",
        "first_byte_timeout_seconds",
        "stream_read_timeout_seconds",
        "request_timeout_seconds",
    }
)


def _route_matches(provider: str | None, model: str | None, base_url: str | None) -> bool:
    try:
        host = (urlparse(str(base_url or "")).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    provider_id = str(provider or "").strip().lower()
    model_text = str(model or "").strip().lower()
    return provider_id == "s3lab" and host == "llm.s3lab.io" and model_text == "qwen3.8-27b"


def _policy_values(raw: object) -> dict:
    if not isinstance(raw, dict):
        return {}
    values: dict = {}
    nested = raw.get("request_policy")
    if isinstance(nested, dict):
        values.update({key: value for key, value in nested.items() if key in _POLICY_KEYS})
    values.update({key: value for key, value in raw.items() if key in _POLICY_KEYS})
    return values


def _config_policy_values(provider: str | None, model: str | None, base_url: str | None) -> dict:
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return {}
    if not isinstance(config, dict):
        return {}
    provider_id = str(provider or "").strip().lower()
    provider_config = None
    provider_configs = config.get("providers")
    if isinstance(provider_configs, dict):
        for key, value in provider_configs.items():
            if str(key).strip().lower() in {provider_id, provider_id.removeprefix("custom:")}:
                provider_config = value
                break
    if provider_config is None:
        custom_entries = config.get("custom_providers")
        route_host = (urlparse(str(base_url or "")).hostname or "").lower().rstrip(".")
        if isinstance(custom_entries, list):
            for entry in custom_entries:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip().lower()
                entry_host = (urlparse(str(entry.get("base_url") or "")).hostname or "").lower().rstrip(".")
                if name in {provider_id, provider_id.removeprefix("custom:")} or entry_host == route_host:
                    provider_config = entry
                    break
    values = _policy_values(provider_config)
    if isinstance(provider_config, dict) and model:
        models = provider_config.get("models")
        if isinstance(models, dict):
            model_key = next((key for key in models if str(key).lower() == str(model).lower()), None)
            if model_key is not None:
                values.update(_policy_values(models[model_key]))
    return values


class S3LabQwenRequestPolicy(ProviderRequestPolicyPlugin):
    name = "s3lab-qwen-request-policy"

    def policy_for_route(self, *, provider, model, base_url):
        if not _route_matches(provider, model, base_url):
            return None
        return self.policy_from_mapping(_config_policy_values(provider, model, base_url))

    def admission_key(self, *, provider, model, base_url):
        del provider
        parsed = urlparse(str(base_url or ""))
        host = (parsed.hostname or "").lower().rstrip(".")
        return (str(model or "").strip().lower(), host)


register_request_policy_plugin(S3LabQwenRequestPolicy())


# Compatibility-only helpers for callers that used the old stream-policy test
# surface. They remain in this plugin so provider-specific names never enter core.
@dataclass(frozen=True)
class QwenStreamBudget:
    total_timeout_seconds: float = 300.0
    idle_timeout_seconds: float = 90.0

    def __post_init__(self) -> None:
        total = float(self.total_timeout_seconds)
        idle = float(self.idle_timeout_seconds)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("total_timeout_seconds must be finite and positive")
        if not math.isfinite(idle) or idle <= 0:
            raise ValueError("idle_timeout_seconds must be finite and positive")
        if idle > total:
            raise ValueError("idle_timeout_seconds cannot exceed total_timeout_seconds")
        object.__setattr__(self, "total_timeout_seconds", total)
        object.__setattr__(self, "idle_timeout_seconds", idle)


@dataclass(frozen=True)
class QwenStreamPolicy:
    budget: QwenStreamBudget = QwenStreamBudget()
    max_concurrency: int = DEFAULT_MAX_CONCURRENT

    @property
    def total_timeout_seconds(self) -> float:
        return self.budget.total_timeout_seconds

    @property
    def idle_timeout_seconds(self) -> float:
        return self.budget.idle_timeout_seconds


class QwenStreamTimeout(TimeoutError):
    def __init__(self, phase: str, *, elapsed_seconds: float):
        if phase not in {"first_byte", "idle", "total"}:
            raise ValueError(f"unknown stream timeout phase: {phase}")
        self.phase = phase
        self.elapsed_seconds = float(elapsed_seconds)
        super().__init__(f"provider stream {phase} timeout after {self.elapsed_seconds:.1f}s")


def budget_expiry(budget: QwenStreamBudget, *, started_at: float, last_progress_at: float | None, now: float) -> str | None:
    elapsed = max(0.0, float(now) - float(started_at))
    if elapsed >= budget.total_timeout_seconds:
        return "total"
    if last_progress_at is None:
        return "first_byte" if elapsed >= budget.idle_timeout_seconds else None
    idle = max(0.0, float(now) - float(last_progress_at))
    return "idle" if idle >= budget.idle_timeout_seconds else None


class QwenStreamAdmissionError(RuntimeError):
    admission_closed = True
    retryable = False

    def __init__(self, model: str, *, active: int, limit: int):
        self.model, self.active, self.limit = model, active, limit
        super().__init__(f"provider admission is full for {model}: {active} active (limit {limit})")


class QwenStreamAdmissionLease:
    def __init__(self, lease):
        self._lease = lease

    @property
    def active(self) -> int:
        return self._lease._owner.snapshot()["active"]

    def close(self, *, reason: str) -> None:
        self._lease.release(reason)


_COMPAT_ADMISSIONS: dict[str, ProviderAdmission] = {}
_COMPAT_LOCK = threading.Lock()


def acquire_qwen_admission(model: str, *, limit: int = DEFAULT_MAX_CONCURRENT) -> QwenStreamAdmissionLease:
    key = str(model or "unknown")
    with _COMPAT_LOCK:
        gate = _COMPAT_ADMISSIONS.setdefault(key, ProviderAdmission(max_concurrent=limit, max_queued=0))
    try:
        return QwenStreamAdmissionLease(gate.acquire(timeout=0))
    except ProviderAdmissionRejected:
        snapshot = gate.snapshot()
        raise QwenStreamAdmissionError(key, active=snapshot["active"], limit=limit) from None


def reset_qwen_admission_for_tests() -> None:
    with _COMPAT_LOCK:
        _COMPAT_ADMISSIONS.clear()


__all__ = [
    "QwenStreamAdmissionError", "QwenStreamAdmissionLease", "QwenStreamBudget", "QwenStreamPolicy",
    "QwenStreamTimeout", "S3LabQwenRequestPolicy", "acquire_qwen_admission", "budget_expiry",
    "reset_qwen_admission_for_tests",
]
