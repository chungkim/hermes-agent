"""Tests for model-provider request-policy plugin discovery and scope."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def test_user_model_provider_plugin_registers_a_generic_policy_in_subprocess(tmp_path):
    home = tmp_path / "home"
    plugin = home / "plugins" / "model-providers" / "route-policy"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: route-policy\nkind: model-provider\nversion: 1.0.0\n", encoding="utf-8"
    )
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


def test_unregistered_routes_keep_core_compatibility_default():
    from agent.provider_request_policy import automatic_policy_for_route

    assert automatic_policy_for_route("default", "any-model", "https://example.invalid/v1") is None


def test_public_discovery_keeps_home_scoped_policy_registrations_isolated(tmp_path):
    homes = {name: tmp_path / name for name in ("home-a", "home-b")}
    for name, home in homes.items():
        plugin_name = f"scope-{name[-1]}"
        route = f"https://{name}.invalid/v1"
        plugin = home / "plugins" / "model-providers" / plugin_name
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(
            f"name: {plugin_name}\nkind: model-provider\nversion: 1.0.0\n", encoding="utf-8"
        )
        (plugin / "__init__.py").write_text(
            "from providers import register_provider\n"
            "from providers.base import ProviderProfile\n"
            "from agent.provider_request_policy import ProviderRequestPolicyPlugin, register_request_policy_plugin\n"
            f"register_provider(ProviderProfile(name='{plugin_name}'))\n"
            "class Policy(ProviderRequestPolicyPlugin):\n"
            f"    name = '{plugin_name}'\n"
            "    def policy_for_route(self, *, provider, model, base_url):\n"
            f"        if base_url == '{route}':\n"
            "            return self.policy_from_mapping({'request_timeout_seconds': 301})\n"
            "        return None\n"
            "register_request_policy_plugin(Policy())\n",
            encoding="utf-8",
        )
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code = """
import json
import os
from agent.provider_request_policy import automatic_policy_for_route
from providers import get_provider_profile, list_providers

home_a, home_b = os.environ["HOME_A"], os.environ["HOME_B"]
assert get_provider_profile("scope-a") is not None
assert automatic_policy_for_route("p", "m", "https://home-a.invalid/v1") is not None
os.environ["HERMES_HOME"] = home_b
assert get_provider_profile("scope-b") is not None
print(json.dumps({
    "b": automatic_policy_for_route("p", "m", "https://home-b.invalid/v1") is not None,
    "a_from_b": automatic_policy_for_route("p", "m", "https://home-a.invalid/v1") is not None,
}))
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(homes["home-a"])
    env["HOME_A"] = str(homes["home-a"])
    env["HOME_B"] = str(homes["home-b"])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo, env.get("PYTHONPATH", "")]))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == {"b": True, "a_from_b": False}
