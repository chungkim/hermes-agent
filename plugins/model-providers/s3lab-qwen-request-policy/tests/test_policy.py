"""Tests for the provider-specific policy plugin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_DEFAULT_PROVIDER_NAMES = {
    "actual", "ai-gateway", "alibaba", "alibaba-cn", "alibaba-coding-plan", "alibaba-coding-plan-cn",
    "alibaba-token-plan", "alibaba-token-plan-cn", "anthropic", "arcee", "azure-foundry", "bedrock",
    "commandcode", "commandcode-anthropic", "copilot", "copilot-acp", "custom", "deepinfra", "deepseek",
    "fireworks", "gemini", "gmi", "huggingface", "kilocode", "kimi-coding", "kimi-coding-cn", "lmstudio",
    "meta-ai", "minimax", "minimax-cn", "minimax-oauth", "nebius-token-factory", "nous", "novita", "nvidia",
    "ollama-cloud", "openai-codex", "opencode-go", "opencode-zen", "openrouter", "qwen-oauth", "router",
    "solstice", "stepfun", "tencent-tokenhub", "upstage", "vertex", "xai", "xiaomi", "zai",
}


def _public_discovery(repo: Path, home: Path) -> dict:
    code = """
import json
from agent.provider_request_policy import automatic_policy_for_route, list_request_policy_plugins
from providers import get_provider_profile, list_providers

names = {profile.name for profile in list_providers()}
assert get_provider_profile("router") is not None
routes = {
    "canonical": automatic_policy_for_route("s3lab", "Qwen3.8-27B", "https://LLM.S3LAB.IO/v1"),
    "vendor_namespace": automatic_policy_for_route("s3lab", "s3lab/Qwen3.8-27B", "https://llm.s3lab.io/v1"),
    "arbitrary_namespace": automatic_policy_for_route("s3lab", "evil/Qwen3.8-27B", "https://llm.s3lab.io/v1"),
    "model_variant": automatic_policy_for_route("s3lab", "Qwen3.8-27B-Instruct", "https://llm.s3lab.io/v1"),
    "provider_mismatch": automatic_policy_for_route("custom", "Qwen3.8-27B", "https://llm.s3lab.io/v1"),
    "host_mismatch": automatic_policy_for_route("s3lab", "Qwen3.8-27B", "https://evil.llm.s3lab.io/v1"),
}
plugins = {plugin.name for plugin in list_request_policy_plugins()}
print(json.dumps({"providers": sorted(names), "routes": {model: value is not None for model, value in routes.items()}, "plugins": sorted(plugins)}))
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo), env.get("PYTHONPATH", "")]))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def test_public_discovery_finds_exact_plugin_without_changing_provider_catalog(tmp_path):
    repo = Path(__file__).resolve().parents[4]
    result = _public_discovery(repo, tmp_path / "isolated-home")
    assert set(result["providers"]) == _DEFAULT_PROVIDER_NAMES
    assert result["routes"] == {
        "canonical": True,
        "vendor_namespace": False,
        "arbitrary_namespace": False,
        "model_variant": False,
        "provider_mismatch": False,
        "host_mismatch": False,
    }
    assert "s3lab-qwen-request-policy" in result["plugins"]
