"""Hermetic Docker llama.cpp local-model mount coverage."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import yaml

import pytest

from tui.backends.llamacpp.command import render_command
from tui.common.profile_store import StoredProfile, _profile_to_entry, _validate_profile


def _profile(directory: Path, **changes) -> StoredProfile:
    values = dict(name="local", backend="llamacpp", runtime="docker",
                  model_volume=str(directory), model_file="model.gguf")
    values.update(changes)
    return StoredProfile(**values)


def test_docker_model_volume_is_persisted_and_renders_container_model(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    profile = _profile(models)
    _validate_profile(profile)
    assert _profile_to_entry(profile)["model_volume"] == str(models)
    command = render_command({}, local_model="/models/model.gguf")
    assert command[command.index("--model") + 1] == "/models/model.gguf"
    assert "-hf" not in command


@pytest.mark.parametrize("filename", ["", "/models/model.gguf", "../model.gguf", "dir/../../model.gguf"])
def test_docker_model_volume_rejects_unsafe_model_file(tmp_path: Path, filename: str) -> None:
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(ValueError, match="model_volume"):
        _validate_profile(_profile(models, model_file=filename))


def test_docker_model_volume_rejects_hf_and_host_modes(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(ValueError, match="Hugging Face"):
        _validate_profile(_profile(models, hf_repo="org/repo"))
    with pytest.raises(ValueError, match="requires llama.cpp runtime=docker"):
        _validate_profile(_profile(models, runtime="host", model_path="model.gguf"))


def test_docker_model_volume_requires_existing_absolute_directory() -> None:
    with pytest.raises(ValueError, match="existing absolute directory"):
        _validate_profile(StoredProfile(name="local", backend="llamacpp", model_volume="relative", model_file="model.gguf"))


def test_override_renders_read_only_model_directory_mount(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (tmp_path / "compose").mkdir()
    (tmp_path / "profiles.example.yaml").write_text("version: 1\n")
    config_dir = tmp_path / "config" / "llamacpp"
    config_dir.mkdir(parents=True)
    (config_dir / "local.yaml").write_text("ctx-size: 1024\n")
    (tmp_path / "profiles.yaml").write_text(yaml.safe_dump({"version": 1, "profiles": [{
        "name": "local", "backend": "llamacpp", "model_volume": str(models), "model_file": "model.gguf",
    }]}))
    script = Path(__file__).parents[1] / "scripts" / "llamacpp" / "render-override.py"
    result = subprocess.run([sys.executable, str(script), "local"], env={**os.environ, "LLMUX_ROOT": str(tmp_path)}, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    override = yaml.safe_load(Path(result.stdout.strip()).read_text())
    service = override["services"]["llama-server"]
    assert service["volumes"] == ["${MODEL_VOLUME_PATH:?MODEL_VOLUME_PATH is required for model_volume}:/models:ro"]
    assert service["command"][service["command"].index("--model") + 1] == "/models/model.gguf"
