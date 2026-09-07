"""Host integration tests use a real detached process, no Docker or model weights."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest
import yaml

from tui.backends.llamacpp.command import render_command
from tui.backends.llamacpp import host_process
from tui.common.profile_store import StoredProfile, _validate_profile

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "compose").mkdir()
    (tmp_path / "config" / "llamacpp").mkdir(parents=True)
    (tmp_path / "config" / "vllm").mkdir(parents=True)
    (tmp_path / "profiles.example.yaml").write_text("version: 1\n")
    model = tmp_path / "tiny model.gguf"
    model.write_bytes(b"GGUF")
    binary = tmp_path / "custom server"
    binary.write_text(f'''#!{sys.executable}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
if "--help" in sys.argv:
    print("--host --port --model --custom-flag --metrics --no-webui")
    sys.exit(0)
print("custom binary running " + os.environ.get("CUSTOM_HOST_ENV", "missing"), flush=True)
if os.environ.get("HOST_TEST_CRASH"):
    sys.exit(7)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if os.environ.get("HOST_TEST_NOT_READY"):
            self.send_response(503); self.end_headers(); return
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({{"data": [{{"id": "tiny"}}]}}).encode())
    def log_message(self, *args): pass
port = int(sys.argv[sys.argv.index("--port") + 1])
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
''')
    binary.chmod(0o700)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    data = {"version": 1, "profiles": [{"name": "local", "backend": "llamacpp",
        "runtime": "host", "server_binary": str(binary), "model_path": str(model),
        "port": port, "env_vars": {"CUSTOM_HOST_ENV": "present"}}]}
    (tmp_path / "profiles.yaml").write_text(yaml.safe_dump(data))
    yield tmp_path
    cli(tmp_path, "down", "local")


def cli(project, *args):
    env = {**os.environ, "LLMUX_ROOT": str(project), "NO_COLOR": "1",
           "PYTHONPATH": str(ROOT), "PATH": "/nonexistent"}
    return subprocess.run([sys.executable, "-m", "tui", *args], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=50)


def assert_ok(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_host_lifecycle_without_docker(project):
    assert_ok(cli(project, "prepare", "local"))
    assert not (project / ".runtime/llamacpp/host/local/state.json").exists()
    assert_ok(cli(project, "up", "local"))
    result = cli(project, "ps", "--runtime", "host", "--json")
    assert_ok(result)
    row = json.loads(result.stdout)[0]
    assert row["running"] and row["ready"] and row["pid"]
    assert row["container"] == "" and row["runtime"] == "host"
    assert_ok(cli(project, "up", "local"))
    again = json.loads(cli(project, "ps", "--runtime", "host", "--json").stdout)[0]
    assert again["pid"] == row["pid"]
    result = cli(project, "logs", "local", "--no-follow")
    assert_ok(result)
    assert "custom binary running present" in result.stdout
    assert cli(project, "profile", "edit", "local", "--port", "19099").returncode != 0
    assert cli(project, "profile", "delete", "local", "--yes").returncode != 0
    assert cli(project, "profile", "rename", "local", "changed").returncode != 0
    assert_ok(cli(project, "profile", "clone", "local", "copy"))
    assert not (project / ".runtime/llamacpp/host/copy/state.json").exists()
    assert_ok(cli(project, "down", "local"))
    assert_ok(cli(project, "down", "local"))
    state = json.loads(cli(project, "ps", "--runtime", "host", "--json").stdout)
    assert not any(row["running"] for row in state)
    assert_ok(cli(project, "logs", "local", "--no-follow"))


def test_host_local_quick_setup_and_flags(project):
    result = cli(project, "config", "flags", "--profile", "local", "--json")
    assert_ok(result)
    assert "custom-flag" in json.loads(result.stdout)["flags"]
    result = cli(project, "profile", "quick-setup", "--backend", "llamacpp", "--runtime", "host",
                 "--name", "quick", "--server-binary", str(project / "custom server"),
                 "--model-path", str(project / "tiny model.gguf"))
    assert_ok(result)
    result = cli(project, "profile", "show", "quick", "--json")
    assert_ok(result)
    assert json.loads(result.stdout)["runtime"] == "host"
    assert_ok(cli(project, "prepare", "quick"))


def test_host_failures_and_mixed_status(project):
    for option in ("--pull", "--dev", "--default-image"):
        assert cli(project, "up", "local", option).returncode != 0
    data = yaml.safe_load((project / "profiles.yaml").read_text())
    data["profiles"][0]["env_vars"]["HOST_TEST_CRASH"] = "1"
    (project / "profiles.yaml").write_text(yaml.safe_dump(data))
    result = cli(project, "up", "local")
    assert result.returncode != 0
    assert_ok(cli(project, "down", "local"))
    data["profiles"].append({"name": "container", "backend": "vllm"})
    (project / "profiles.yaml").write_text(yaml.safe_dump(data))
    result = cli(project, "ps", "--json")
    assert result.returncode != 0
    rows = json.loads(result.stdout)
    assert next(r for r in rows if r["profile"] == "container")["status"] == "unknown"
    assert next(r for r in rows if r["profile"] == "local")["runtime"] == "host"
    assert_ok(cli(project, "ps", "--runtime", "host", "--json"))


def test_renderer_preserves_input_and_handles_paths():
    cfg = {"ctx-size": 1024, "extra-args": ["--custom-flag", "a b"], "flash-attn": True}
    before = dict(cfg)
    argv = render_command(cfg, local_model="/models/a b.gguf", host="127.0.0.1", port=8123)
    assert cfg == before
    assert argv[:4] == ["--host", "127.0.0.1", "--port", "8123"]
    assert argv[argv.index("--model") + 1] == "/models/a b.gguf"
    assert argv[-2:] == ["--custom-flag", "a b"]
    for flag in ("--port=9000", "-m", "--model=x", "-hf", "--hf-repo"):
        with pytest.raises(ValueError):
            render_command({"extra-args": [flag]}, local_model="x.gguf")


def test_pid_identity_never_signals_reused_pid():
    record = host_process.identity(os.getpid())
    assert host_process.alive(record)
    wrong = {**record, "start": "0"}
    assert not host_process.alive(wrong)
    assert not host_process.send_signal(wrong, signal.SIGTERM)


def test_profile_validation():
    with pytest.raises(ValueError, match="vLLM"):
        _validate_profile(StoredProfile(name="x", backend="vllm", runtime="host"))
    with pytest.raises(ValueError, match="OR"):
        _validate_profile(StoredProfile(name="x", backend="llamacpp", runtime="host", model_path="x", hf_repo="a/b"))
    with pytest.raises(ValueError, match="require"):
        _validate_profile(StoredProfile(name="x", backend="llamacpp", model_path="x"))


def test_local_shards_and_offline_hf_prepare(project):
    profile_file = project / "profiles.yaml"
    data = yaml.safe_load(profile_file.read_text())
    profile = data["profiles"][0]
    profile["model_path"] = str(project / "split-00001-of-00002.gguf")
    Path(profile["model_path"]).write_bytes(b"GGUF")
    profile_file.write_text(yaml.safe_dump(data))
    assert cli(project, "prepare", "local").returncode != 0
    (project / "split-00002-of-00002.gguf").write_bytes(b"GGUF")
    assert_ok(cli(project, "prepare", "local"))
    cache = project / "hf-cache"
    repo = cache / "models--test--tiny"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "tiny.gguf").write_bytes(b"GGUF")
    (repo / "refs").mkdir()
    (repo / "refs/main").write_text("a" * 40)
    profile.pop("model_path")
    profile.update(hf_repo="test/tiny", hf_file="tiny.gguf")
    profile["env_vars"].update(HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE="1")
    profile_file.write_text(yaml.safe_dump(data))
    assert_ok(cli(project, "prepare", "local"))
    assert_ok(cli(project, "up", "local"))


def test_env_check_host_without_common_env(project):
    assert_ok(cli(project, "env-check", "--profile", "local", "--json"))
    assert_ok(cli(project, "env-check", "--profile", "local"))


@pytest.mark.asyncio
async def test_host_forms_and_start_screen(monkeypatch, tmp_path):
    from textual.app import App
    from textual.widgets import Input, Select
    from tui.backends.llamacpp.backend import Profile
    from tui.backends.llamacpp.screens import profile as profile_screen
    from tui.backends.llamacpp.screens import container as container_screen
    from tui.backends.llamacpp.screens import quick_setup as quick_screen
    from unittest.mock import AsyncMock
    profile = Profile(name="local", runtime="host", model_path="/tmp/tiny.gguf")
    monkeypatch.setattr(profile_screen, "list_config_names", lambda: [])
    monkeypatch.setattr(container_screen.backend, "load_profile", lambda _: profile)
    docker_probe = AsyncMock(side_effect=AssertionError("host screen must not probe Docker"))
    monkeypatch.setattr(container_screen, "list_local_dev_images", docker_probe)
    app = App()
    async with app.run_test(size=(120, 50)) as pilot:
        form = profile_screen.ProfileFormScreen(profile)
        await app.push_screen(form)
        await pilot.pause()
        assert form.query_one("#runtime-select", Select).value == "host"
        assert form.query_one("#binary-row").display
        assert not form.query_one("#image-tag-input", Input).parent.display
        app.pop_screen()
        startup = container_screen.ContainerUpScreen("local")
        await app.push_screen(startup)
        await pilot.pause()
        assert not startup.query_one("#version-scroll").display
        docker_probe.assert_not_called()
        app.pop_screen()
        quick = quick_screen.QuickSetupScreen()
        await app.push_screen(quick)
        await pilot.pause()
        assert quick.query_one("#runtime-select", Select).value == "docker"
        assert quick.query_one("#model-path-input", Input).value == ""


def run_python(project, source):
    return subprocess.run([sys.executable, "-c", source], cwd=ROOT,
        env={**os.environ, "LLMUX_ROOT": str(project), "PYTHONPATH": str(ROOT), "PATH": "/nonexistent"},
        capture_output=True, text=True, timeout=50)


def test_readiness_timeout_and_cancel_keep_owned_server(project):
    data = yaml.safe_load((project / "profiles.yaml").read_text())
    data["profiles"][0]["env_vars"]["HOST_TEST_NOT_READY"] = "1"
    (project / "profiles.yaml").write_text(yaml.safe_dump(data))
    result = run_python(project, '''
import asyncio
from tui.backends.llamacpp.host_runtime import stream_up
async def main():
    events = [event async for event in stream_up("local", timeout=0.1)]
    assert events[-1] == ("rc", 1), events
    assert "remains running" in events[-2][1], events
asyncio.run(main())
''')
    assert_ok(result)
    row = json.loads(cli(project, "ps", "--runtime", "host", "--json").stdout)[0]
    assert row["running"] and not row["ready"]
    assert_ok(cli(project, "down", "local"))
    result = run_python(project, '''
import asyncio
from tui.backends.llamacpp.host_runtime import stream_up
async def main():
    stream = stream_up("local", timeout=10)
    async for kind, value in stream:
        if kind == "log" and "Host process started" in value:
            await stream.aclose()
            break
asyncio.run(main())
''')
    assert_ok(result)
    row = json.loads(cli(project, "ps", "--runtime", "host", "--json").stdout)[0]
    assert row["running"]


def test_concurrent_launch_and_forced_stop(project):
    binary = project / "custom server"
    source = binary.read_text().replace('import json, os, sys',
        'import json, os, sys, signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)')
    binary.write_text(source)
    result = run_python(project, '''
import asyncio
from tui.backends.llamacpp.host_runtime import stream_up
async def launch():
    return [e async for e in stream_up("local")]
async def main():
    a, b = await asyncio.gather(launch(), launch())
    assert any(events[-1] == ("rc", 0) for events in (a, b)), (a, b)
asyncio.run(main())
''')
    assert_ok(result)
    started = time.monotonic()
    assert_ok(cli(project, "down", "local"))
    assert time.monotonic() - started >= 29
    row = json.loads(cli(project, "ps", "--runtime", "host", "--json").stdout)[0]
    assert not row["running"] and row["exit_code"] == -signal.SIGKILL


def test_defaults_and_local_profile_creation(project):
    data = yaml.safe_load((project / "profiles.yaml").read_text())
    data["defaults"] = {"llamacpp": {"runtime": "host", "server_binary": str(project / "custom server")}}
    (project / "profiles.yaml").write_text(yaml.safe_dump(data))
    assert_ok(cli(project, "profile", "new", "inherited", "--backend", "llamacpp",
                  "--model-path", str(project / "tiny model.gguf")))
    shown = cli(project, "profile", "show", "inherited", "--json")
    assert_ok(shown)
    assert json.loads(shown.stdout)["runtime"] == "host"
    assert_ok(cli(project, "prepare", "inherited"))


@pytest.mark.asyncio
async def test_log_tail_rotation_and_stale_state(tmp_path, monkeypatch):
    from tui.common import profile_store
    monkeypatch.setattr(profile_store, "RUNTIME_DIR", tmp_path)
    folder = host_process.ensure_directory("logs")
    host_process.write_state("logs", {"version": 1, "status": "running",
                                     "server": {"pid": os.getpid(), "start": "0", "boot": "old"}})
    assert not host_process.status("logs")["active"]
    (folder / "server.log.1").write_text("one\ntwo\n")
    (folder / "server.log").write_text("three\nfour\n")
    assert [line async for line in host_process.logs("logs", tail=3, follow=False)] == ["two", "three", "four"]
    assert [line async for line in host_process.logs("logs", tail=0, follow=False)] == []
    (folder / "state.json").write_text("broken")
    with pytest.raises(RuntimeError, match="Cannot read"):
        host_process.status("logs")
