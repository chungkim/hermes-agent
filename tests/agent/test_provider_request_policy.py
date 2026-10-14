"""Focused tests for the generic request-policy seam."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import pytest


def test_user_model_provider_plugin_registers_a_generic_policy_in_subprocess(tmp_path):
    home = tmp_path / "home"
    plugin = home / "plugins" / "model-providers" / "route-policy"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: route-policy\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "from agent.provider_request_policy import ProviderRequestPolicyPlugin, register_request_policy_plugin\n"
        "register_provider(ProviderProfile(name='route-policy'))\n"
        "class TestPolicy(ProviderRequestPolicyPlugin):\n"
        "    name = 'test-policy'\n"
        "    def policy_for_route(self, *, provider, model, base_url):\n"
        "        if (provider, model, base_url) == ('test-provider', 'test-model', 'https://test.invalid/v1'):\n"
        "            return self.policy_from_mapping({'request_timeout_seconds': 300})\n"
        "        return None\n"
        "register_request_policy_plugin(TestPolicy())\n",
        encoding="utf-8",
    )
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code = """
import json
from agent.provider_request_policy import automatic_policy_for_route, list_request_policy_plugins
from providers import get_provider_profile, list_providers

names = {profile.name for profile in list_providers()}
assert get_provider_profile("route-policy") is not None
policy = automatic_policy_for_route("test-provider", "test-model", "https://test.invalid/v1")
plugins = {plugin.name for plugin in list_request_policy_plugins()}
print(json.dumps({"names": sorted(names), "timeout": policy.total_timeout if policy else None, "plugins": sorted(plugins)}))
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo, env.get("PYTHONPATH", "")]))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True, check=True
    )
    payload = json.loads(result.stdout)
    assert "route-policy" in payload["names"]
    assert payload["timeout"] == 300.0
    assert "test-policy" in payload["plugins"]


def test_unregistered_routes_keep_legacy_compatibility_default():
    from agent.provider_request_policy import automatic_policy_for_route

    assert automatic_policy_for_route("default", "any-model", "https://example.invalid/v1") is None


class TestProviderAdmission:
    def test_rejects_when_route_is_full_and_queue_is_disabled(self):
        from agent.provider_request_policy import ProviderAdmission, ProviderAdmissionRejected

        admission = ProviderAdmission(max_concurrent=1, max_queued=0)
        first = admission.acquire(timeout=0)
        try:
            with pytest.raises(ProviderAdmissionRejected, match="capacity"):
                admission.acquire(timeout=0)
        finally:
            first.release("test_complete")

    def test_waiter_is_admitted_after_slot_release_and_cleanup_is_idempotent(self):
        from agent.provider_request_policy import ProviderAdmission

        admission = ProviderAdmission(max_concurrent=1, max_queued=1)
        first = admission.acquire(timeout=0)
        outcome = {}

        def waiter():
            outcome["lease"] = admission.acquire(timeout=1)

        thread = threading.Thread(target=waiter)
        thread.start()
        deadline = time.monotonic() + 1
        while admission.snapshot()["waiting"] != 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        first.release("test_release")
        thread.join(1)
        outcome["lease"].release("test_complete")
        outcome["lease"].release("duplicate_release")
        assert admission.snapshot()["active"] == 0
        assert admission.snapshot()["waiting"] == 0

    def test_context_manager_releases_after_exception(self):
        from agent.provider_request_policy import ProviderAdmission

        admission = ProviderAdmission(max_concurrent=1, max_queued=0)
        with pytest.raises(RuntimeError):
            with admission.lease(timeout=0):
                raise RuntimeError("stream cleanup")
        assert admission.snapshot()["active"] == 0


class TestPolicyContract:
    def test_validation_and_derived_deadlines_are_bounded(self):
        from agent.provider_request_policy import (
            MAX_FIRST_BYTE_TIMEOUT,
            MAX_IDLE_TIMEOUT,
            MAX_QUEUE_TIMEOUT,
            MAX_TOTAL_TIMEOUT,
            ProviderRequestPolicy,
        )

        with pytest.raises(ValueError, match="max_concurrent"):
            ProviderRequestPolicy.from_mapping({"max_concurrent_requests": 0})
        selected = ProviderRequestPolicy.from_mapping({}).with_derived_budgets(
            context_tokens=10**9,
            admission_position=10**9,
            observed_queue_wait=10**9,
        )
        assert selected.queue_timeout <= MAX_QUEUE_TIMEOUT
        assert selected.first_byte_timeout <= MAX_FIRST_BYTE_TIMEOUT
        assert selected.idle_timeout <= MAX_IDLE_TIMEOUT
        assert selected.total_timeout <= MAX_TOTAL_TIMEOUT

    def test_deadline_and_admission_errors_are_terminal(self):
        from agent.error_classifier import classify_api_error
        from agent.provider_request_policy import (
            ProviderAdmissionRejected,
            ProviderRequestDeadlineExceeded,
        )

        for error in (ProviderAdmissionRejected("capacity"), ProviderRequestDeadlineExceeded("deadline")):
            classified = classify_api_error(error, provider="custom", model="model")
            assert classified.retryable is False
            assert classified.should_fallback is False


def test_error_surface_preserves_terminal_deadline_metadata():
    from agent.error_classifier import classify_api_error
    from agent.error_surface import build_error_surface_from_result
    from agent.provider_request_policy import ProviderRequestDeadlineExceeded

    error = ProviderRequestDeadlineExceeded("selected total budget expired")
    classified = classify_api_error(error, provider="custom", model="model")
    result = {
        "error": str(error),
        "failure_retryable": classified.retryable,
        "provider_request_deadline": True,
        "failure_reason": classified.reason.value,
    }
    surface = build_error_surface_from_result(result, provider="custom", model="model")
    assert surface is not None
    assert surface["code"] == "request_deadline_exceeded"
    assert surface["retryable"] is False


def test_partial_output_deadline_stays_terminal_instead_of_becoming_a_stub(monkeypatch):
    """Delivered output must not turn a consumed deadline into a replay."""
    import agent.chat_completion_helpers as helpers
    from agent.provider_request_policy import ProviderRequestDeadlineExceeded

    error = ProviderRequestDeadlineExceeded("selected total budget expired")
    call = object.__new__(helpers._StreamingCall)
    call.agent = type("Agent", (), {
        "_interrupt_requested": False,
        "_current_streamed_assistant_text": "partial tool-call text",
    })()
    call.result = {"response": None, "error": error, "partial_tool_names": ["write_file"]}
    call.deltas_were_sent = {"yes": True}
    call._request_cancelled = {"value": True}
    call._admission_lease = None
    call._resolve_request_budget = lambda: None
    call._acquire_admission = lambda: None
    call._resolve_stale_timeout = lambda: None
    call._run_call = lambda: None
    call._monitor_loop = lambda: None
    call._partial_stream_stub = lambda: pytest.fail("deadline must not become a continuation stub")
    call._stream_timeout_error = error
    monkeypatch.setattr(helpers, "should_use_direct_api_call", lambda _agent: True)

    with pytest.raises(ProviderRequestDeadlineExceeded) as raised:
        call.run()
    assert raised.value is error
    assert raised.value.provider_request_deadline is True
    assert raised.value.retryable is False


def test_total_deadline_cancellation_is_idempotent_and_cleans_up_once():
    import agent.chat_completion_helpers as helpers
    from types import SimpleNamespace

    call = object.__new__(helpers._StreamingCall)
    call._stream_timeout_error = None
    call._request_cancelled = {"value": False}
    call._attempt_stream_response = None
    call.clients = SimpleNamespace(diag={}, close_once=lambda *_: None)  # type: ignore[assignment]
    call.agent = SimpleNamespace(model="test-model")
    cancelled = []
    call._cancel_current_stream_attempt = lambda reason: cancelled.append(reason)
    call._shutdown_stale_attempt_socket = lambda response: None
    call._kill_total_stream(300.0)
    call._kill_total_stream(301.0)
    assert call._request_cancelled["value"] is True
    assert cancelled == ["provider_total_timeout"]
    assert call.clients.diag["stream_timeout_phase"] == "total"
