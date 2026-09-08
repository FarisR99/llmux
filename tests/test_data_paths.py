"""Isolated coverage for code/data-root resolution and legacy migration."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkout(root: Path, *, active: bool = False) -> None:
    (root / "compose").mkdir(parents=True)
    (root / "profiles.example.yaml").write_text("version: 1\n")
    (root / "config" / "vllm").mkdir(parents=True)
    (root / "profiles.yaml").write_text(yaml.safe_dump({"version": 1, "profiles": [
        {"name": "demo", "backend": "vllm", "model_id": "org/model"},
    ]}))
    (root / ".env.common").write_text("HF_TOKEN=never-print-this\n")
    (root / "config" / "vllm" / "demo.yaml").write_text("model: org/model\n")
    if active:
        state = root / ".runtime" / "llamacpp" / "host" / "hosted"
        state.mkdir(parents=True)
        (state / "state.json").write_text('{"status":"running"}')


def _cli(root: Path, data: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "LLMUX_ROOT": str(root), "NO_COLOR": "1"}
    if data is not None:
        env["LLMUX_DATA_DIR"] = str(data)
    return subprocess.run([sys.executable, "-m", "tui", *args], cwd=REPO_ROOT,
                          env=env, text=True, capture_output=True, timeout=20)


def test_explicit_data_root_and_migration_dry_run(tmp_path: Path) -> None:
    source, data = tmp_path / "legacy", tmp_path / "data"
    _checkout(source)
    result = _cli(source, data, "data", "status", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)["result"]
    assert payload["code_root"] == str(source)
    assert payload["data_root"] == str(data)
    assert payload["data_source"] == "LLMUX_DATA_DIR"
    assert payload["migration_ready"] is True
    plan = _cli(source, data, "data", "migrate", "--json")
    assert plan.returncode == 0
    assert json.loads(plan.stdout)["result"]["copied"] is False
    assert not data.exists()


def test_migration_copies_verified_state_without_runtime_or_secret_output(tmp_path: Path) -> None:
    source, data = tmp_path / "legacy", tmp_path / "data"
    _checkout(source)
    result = _cli(source, data, "data", "migrate", "--execute", "--yes", "--json")
    assert result.returncode == 0
    body = json.loads(result.stdout)["result"]
    assert body["copied"] is True
    assert "never-print-this" not in result.stdout + result.stderr
    assert (data / "profiles.yaml").read_bytes() == (source / "profiles.yaml").read_bytes()
    assert (data / "config" / "vllm" / "demo.yaml").exists()
    assert not (data / ".runtime").exists()
    assert oct((data / ".env.common").stat().st_mode & 0o777) == "0o600"


def test_migration_refuses_active_or_conflicting_state(tmp_path: Path) -> None:
    active, destination = tmp_path / "active", tmp_path / "data"
    _checkout(active, active=True)
    refused = _cli(active, destination, "data", "migrate", "--json")
    assert refused.returncode == 1
    assert json.loads(refused.stdout)["error"]["code"] == "ACTIVE_PROFILES"
    source, nonempty = tmp_path / "legacy", tmp_path / "nonempty"
    _checkout(source)
    nonempty.mkdir()
    (nonempty / "other").write_text("x")
    refused = _cli(source, nonempty, "data", "migrate", "--json")
    assert refused.returncode == 1
    assert json.loads(refused.stdout)["error"]["code"] == "DESTINATION_NOT_EMPTY"


def test_legacy_checkout_defaults_data_to_code_root(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    _checkout(source)
    result = _cli(source, None, "data", "status", "--json")
    payload = json.loads(result.stdout)["result"]
    assert payload["data_root"] == str(source)
    assert payload["data_source"] == "legacy-checkout"


def test_backend_paths_use_data_for_state_and_code_for_compose(tmp_path: Path) -> None:
    source, data = tmp_path / "legacy", tmp_path / "data"
    _checkout(source)
    env = {**os.environ, "LLMUX_ROOT": str(source), "LLMUX_DATA_DIR": str(data),
           "PYTHONPATH": str(REPO_ROOT)}
    code = (
        "from tui.backends.vllm import backend_common as v; "
        "from tui.backends.llamacpp import backend as l; "
        "import json; print(json.dumps({'v_config':str(v.CONFIG_DIR),'v_compose':str(v.COMPOSE_DIR),"
        "'l_runtime':str(l.RUNTIME_DIR),'l_scripts':str(l.SCRIPTS_DIR)}))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout)
    assert paths["v_config"].startswith(str(data))
    assert paths["l_runtime"].startswith(str(data))
    assert paths["v_compose"].startswith(str(source))
    assert paths["l_scripts"].startswith(str(source))


def test_non_checkout_invocation_uses_default_user_data_dir(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "HOME": str(tmp_path)}
    env.pop("LLMUX_ROOT", None)
    env.pop("LLMUX_DATA_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", "from tui.common.paths import PATHS; print(PATHS.data_root)"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / ".local" / "share" / "llmux")
