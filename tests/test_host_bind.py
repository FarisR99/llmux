"""Hermetic host-runtime bind-address coverage."""
from __future__ import annotations

import pytest

from tui.backends.llamacpp.backend_runtime import local_api_url
from tui.backends.llamacpp.command import render_command
from tui.cli.profile import _profile_detail
from tui.backends.llamacpp.backend_runtime import ContainerStatus
from tui.common.profile_store import StoredProfile, _to_profile, _validate_profile


def test_legacy_host_profile_defaults_to_loopback() -> None:
    profile = _to_profile({"name": "legacy", "backend": "llamacpp", "runtime": "host", "model_path": "model.gguf"})
    assert profile.host_bind == "127.0.0.1"


@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0", "::1", "::", "192.0.2.10"])
def test_host_bind_accepts_literal_ip_addresses(bind: str) -> None:
    _validate_profile(StoredProfile(name="host", backend="llamacpp", runtime="host", model_path="model.gguf", host_bind=bind))


@pytest.mark.parametrize("bind", ["example.org", "http://127.0.0.1", "127.0.0.1:8080", "--metrics", "127.0.0.1 bad"])
def test_host_bind_rejects_non_address_or_argument_injection(bind: str) -> None:
    with pytest.raises(ValueError, match="host_bind"):
        _validate_profile(StoredProfile(name="host", backend="llamacpp", runtime="host", model_path="model.gguf", host_bind=bind))


def test_host_bind_is_injected_into_host_command_only() -> None:
    command = render_command({}, local_model="model.gguf", host="0.0.0.0", port=8123)
    assert command[:4] == ["--host", "0.0.0.0", "--port", "8123"]
    docker = StoredProfile(name="docker", backend="llamacpp", runtime="docker", host_bind="127.0.0.1")
    _validate_profile(docker)
    with pytest.raises(ValueError, match="requires runtime=host"):
        _validate_profile(StoredProfile(name="docker", backend="llamacpp", runtime="docker", host_bind="0.0.0.0"))


@pytest.mark.parametrize(("bind", "expected"), [
    ("0.0.0.0", "http://127.0.0.1:8080/v1"),
    ("::", "http://[::1]:8080/v1"),
    ("::1", "http://[::1]:8080/v1"),
    ("127.0.0.1", "http://127.0.0.1:8080/v1"),
])
def test_readiness_url_never_targets_wildcard(bind: str, expected: str) -> None:
    assert local_api_url(8080, bind) == expected


def test_profile_json_and_status_have_bind_address_fields() -> None:
    profile = StoredProfile(name="host", backend="llamacpp", runtime="host", model_path="model.gguf", host_bind="0.0.0.0")
    assert _profile_detail(profile)["host_bind"] == "0.0.0.0"
    assert ContainerStatus(profile_name="host", container_name="", runtime="host", bind_host="0.0.0.0").bind_host == "0.0.0.0"
