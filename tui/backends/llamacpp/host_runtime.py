"""Host llama.cpp preparation and lifecycle, shared by CLI and TUI."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from tui.common import prepare, profile_store
from tui.common.env import expand_env_values, parse_env_file
from . import host_process
from .command import render_command


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = profile_store.PROJECT_ROOT / path
    return Path(os.path.abspath(path))


def environment(profile) -> dict[str, str]:
    env = os.environ.copy()
    common_path = profile_store.PROJECT_ROOT / ".env.common"
    if common_path.exists() and common_path.stat().st_mode & 0o077:
        raise ValueError(".env.common permissions must be owner-only (0600)")
    common = expand_env_values(parse_env_file(common_path))
    for key in ("HF_TOKEN", "HF_ENDPOINT", "TZ"):
        if common.get(key):
            env[key] = common[key]
    env.update(profile.env_vars)
    env["CUDA_VISIBLE_DEVICES"] = profile.gpu_id
    if common.get("HF_CACHE_PATH"):
        env["HF_HOME"] = str(resolve_path(common["HF_CACHE_PATH"]))
        env["HF_HUB_CACHE"] = str(Path(env["HF_HOME"]) / "hub")
    return env


def executable(profile, env: dict[str, str] | None = None) -> str:
    host_process.require_linux()
    value = profile.server_binary
    env = env if env is not None else environment(profile)
    if "/" in value or value.startswith("~"):
        path = str(resolve_path(value))
    else:
        path = shutil.which(value, path=env.get("PATH")) or ""
    if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"llama-server executable not found or not executable: {value}")
    return str(Path(path).resolve())


def model_paths(profile, cfg: dict) -> tuple[str, list[str]]:
    if profile.model_path:
        path = resolve_path(profile.model_path)
        shards = prepare.gguf_shard_names(path.name)
        for shard in shards:
            candidate = path.parent / shard
            if not candidate.is_file() or not os.access(candidate, os.R_OK):
                raise ValueError(f"GGUF file missing or unreadable: {candidate}")
        return str(path.parent / shards[0]), []
    filename = profile.hf_file or cfg.get("model-file") or profile.model_file
    if not profile.hf_repo or not filename:
        raise ValueError("Host profile needs model_path or hf_repo and hf_file/model_file")
    shards = prepare.gguf_shard_names(filename)
    return shards[0], shards


def bundle_paths(profile) -> dict[str, str]:
    """Validate selected local companion assets before launching the server."""
    values = {
        "mmproj": profile.mmproj_path,
        "draft_model": profile.draft_model_path,
        "chat_template": profile.chat_template_path,
    }
    for role, raw_path in values.items():
        if not raw_path:
            continue
        path = resolve_path(raw_path)
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ValueError(f"{role} file missing or unreadable: {path}")
        suffix = ".jinja" if role == "chat_template" else ".gguf"
        if path.suffix.lower() != suffix:
            raise ValueError(f"{role} must be a {suffix} file: {path}")
        values[role] = str(path)
    values["draft_type"] = profile.draft_type
    values["mtp_n"] = profile.mtp_n
    return values


async def _download(profile, filename: str, shards: list[str], env: dict, max_workers):
    workers = max_workers if max_workers is not None else prepare.prepare_max_workers()
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError("max_workers must be a positive integer")
    # A subprocess gives cancellation ownership over downloader threads as well.
    spec = {"repo": profile.hf_repo, "file": filename, "shards": shards, "workers": workers}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tui.backends.llamacpp.host_download",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=env, cwd=str(profile_store.PROJECT_ROOT))
    try:
        output, errors = await proc.communicate(json.dumps(spec).encode())
        if proc.returncode:
            raise RuntimeError(f"Hugging Face download failed (exit {proc.returncode}): {errors.decode(errors='replace')[-2000:]}")
        result = json.loads(output)
        return str(Path(result["model"]))
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def prepare_profile(name: str, *, max_workers: int | None = None) -> dict:
    from .backend import load_config, load_profile
    from .backend_runtime import _ensure_profile_config
    stored = profile_store.load_profile(name, "llamacpp")
    if stored is None or stored.runtime != "host":
        raise ValueError(f"Host profile '{name}' not found")
    env = environment(stored)
    binary = executable(stored, env)
    ok, messages = _ensure_profile_config(stored, load_profile(name))
    if not ok:
        raise ValueError("\n".join(messages))
    stored = profile_store.load_profile(name, "llamacpp")
    cfg = load_config(stored.config_name or name).params
    filename, shards = model_paths(stored, cfg)
    bundle = bundle_paths(stored)
    # Validate arguments before potentially expensive downloads.
    render_command(
        cfg, local_model=filename, host="127.0.0.1", port=stored.port,
        **bundle,
    )
    model = await _download(stored, filename, shards, env, max_workers) if shards else filename
    argv = [binary, *render_command(
        cfg, local_model=model, host="127.0.0.1", port=stored.port,
        **bundle,
    )]
    profile_store.render_env(stored)
    return {"argv": argv, "env": env, "port": stored.port,
            "cwd": str(profile_store.PROJECT_ROOT), "profile": stored, "config": cfg}


async def stream_prepare(name: str, *, max_workers: int | None = None):
    try:
        yield ("log", f"Preparing host profile '{name}' (binary, config, and GGUF weights)")
        await prepare_profile(name, max_workers=max_workers)
        yield ("log", f"Host profile '{name}' prepared. Start with: llmux up {name}")
        yield ("rc", 0)
    except (OSError, ValueError, RuntimeError) as exc:
        yield ("log", f"Error: {exc}")
        yield ("rc", 1)


def port_conflict(profile) -> str | None:
    state = host_process.status(profile.name)
    if state["active"]:
        return None
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", profile.port))
        except OSError as exc:
            import errno
            if exc.errno == errno.EADDRINUSE:
                return f"another local process on 127.0.0.1:{profile.port}"
            raise RuntimeError(f"Port probe failed: {exc}") from exc
    return None


async def conflict_warnings(profile) -> list[str]:
    from tui.common import docker
    warnings = []
    conflict = port_conflict(profile)
    if conflict:
        warnings.append(f"port conflict: {conflict}")
    from .backend import load_config
    if host_process.status(profile.name)["active"] or load_config(profile.config_name or profile.name).get("n-gpu-layers") == 0:
        return warnings
    for other in profile_store.list_profiles("llamacpp"):
        if other.name != profile.name and other.runtime == "host" and host_process.status(other.name)["active"]:
            if load_config(other.config_name or other.name).get("n-gpu-layers") != 0 and set(other.gpu_id.split(",")) & set(profile.gpu_id.split(",")):
                warnings.append(f"GPU conflict: host profile '{other.name}' uses GPU {other.gpu_id}")
    docker_profiles = [p for bk in ("vllm", "llamacpp") for p in profile_store.list_profiles(bk)
                       if p.runtime == "docker"]
    if docker_profiles:
        try:
            running = await docker.running_container_names()
        except RuntimeError as exc:
            # Optional ownership attribution cannot make a host runtime depend on Docker.
            import logging
            logging.getLogger(__name__).warning("Docker GPU overlap check unavailable: %s", exc)
        else:
            for other in docker_profiles:
                if (other.container_name or other.name) in running and set(other.gpu_id.split(",")) & set(profile.gpu_id.split(",")):
                    warnings.append(f"GPU conflict: Docker profile '{other.name}' uses GPU {other.gpu_id}")
    return warnings


async def stream_up(name: str, *, timeout: float = 45.0, max_wait: float = 1800.0):
    from .backend_runtime import _models_endpoint_ready
    from .backend import load_config
    try:
        if host_process.status(name)["active"]:
            state = host_process.status(name)
            if state["running"] and await _models_endpoint_ready(state["port"]):
                yield ("log", f"Host profile '{name}' is already running")
                yield ("rc", 0)
            else:
                yield ("log", f"Host profile '{name}' is active but not ready; inspect logs or stop it")
                yield ("rc", 1)
            return
        yield ("log", f"Preparing host profile '{name}'")
        spec = await prepare_profile(name)
        with host_process.lifecycle_lock(name):
            if host_process.status(name)["active"]:
                raise RuntimeError("Host profile was started concurrently; retry up")
            # Shared storage lock makes snapshot verification + process launch atomic
            # with profile edits; lifecycle lock excludes subsequent edits until registered.
            with profile_store.storage_transaction():
                current = profile_store.load_profile(name, "llamacpp")
                if current != spec["profile"] or load_config(current.config_name or name).params != spec["config"]:
                    raise RuntimeError("Profile/config changed during preparation; retry up")
                conflict = port_conflict(current)
                if conflict:
                    raise RuntimeError(f"Port conflict: {conflict}")
                payload = {key: spec[key] for key in ("argv", "env", "cwd", "port")}
                # The supervisor receives secrets over a pipe, never in argv or state.
                proc = subprocess.Popen(
                    [sys.executable, "-m", "tui.backends.llamacpp.host_process", name],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    cwd=spec["cwd"], start_new_session=True)
                proc.stdin.write(json.dumps(payload).encode())
                proc.stdin.close()
            # Wait for ownership registration before releasing the lifecycle lock.
            registered = False
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    state = host_process.read_state(name)
                    if state.get("supervisor", {}).get("pid") == proc.pid and (state.get("server") or state.get("status") == "failed"):
                        registered = bool(state.get("server"))
                        break
                    if proc.poll() is not None:
                        break
                    await asyncio.sleep(0.05)
                if not registered:
                    raise RuntimeError("Host supervisor failed to launch; inspect host logs")
            finally:
                if not registered:
                    if proc.poll() is None:
                        proc.terminate()
                    await asyncio.to_thread(proc.wait)
                else:
                    # Reap when the interface stays open, without keeping it alive.
                    import threading
                    threading.Thread(target=proc.wait, daemon=True).start()
        yield ("log", f"Host process started (PID {state['server']['pid']}); checking readiness")
        started = time.monotonic()
        deadline = started + timeout
        last_log = None
        last_notice = started
        while time.monotonic() < min(deadline, started + max_wait):
            log_path = host_process.directory(name) / "server.log"
            try:
                stat = log_path.stat()
                log_progress = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            except FileNotFoundError:
                log_progress = None
            if last_log is not None and log_progress != last_log:
                deadline = time.monotonic() + timeout
            last_log = log_progress
            if time.monotonic() - last_notice >= 10:
                yield ("log", "Waiting for host /v1/models; model is still loading")
                last_notice = time.monotonic()
            state = host_process.status(name)
            if not state["running"]:
                raise RuntimeError(f"Host server exited (exit {state.get('exit_code')}); see llmux logs {name}")
            if await _models_endpoint_ready(spec["port"]):
                yield ("log", f"Ready: http://127.0.0.1:{spec['port']}/v1")
                yield ("rc", 0)
                return
            await asyncio.sleep(0.5)
        raise RuntimeError(f"Host server not ready within {timeout:g}s; it remains running. See llmux logs {name}")
    except (OSError, ValueError, RuntimeError) as exc:
        yield ("log", f"Error: {exc}")
        yield ("rc", 1)


async def down(name: str) -> tuple[int, str]:
    try:
        with host_process.lifecycle_lock(name):
            state = host_process.status(name)
            if not state["active"]:
                return 0, f"Host profile '{name}' is already stopped"
            host_process.send_signal(state.get("supervisor"), signal.SIGTERM)
            host_process.send_signal(state.get("server"), signal.SIGTERM)
            deadline = time.monotonic() + 32
            while time.monotonic() < deadline:
                if not host_process.status(name)["active"]:
                    return 0, f"Stopped host profile '{name}'"
                await asyncio.sleep(0.1)
            host_process.send_signal(state.get("server"), signal.SIGKILL)
            await asyncio.sleep(0.5)
            host_process.send_signal(state.get("supervisor"), signal.SIGKILL)
            await asyncio.sleep(0.1)
            if host_process.status(name)["active"]:
                raise RuntimeError("Host processes did not terminate")
            return 0, f"Stopped host profile '{name}'"
    except (OSError, ValueError, RuntimeError) as exc:
        return 1, str(exc)


async def get_status(profile):
    from .backend_runtime import ContainerStatus, _models_endpoint_ready
    state = host_process.status(profile.name)
    ready = state["running"] and await _models_endpoint_ready(state["port"])
    return ContainerStatus(
        profile_name=profile.name, container_name="", runtime="host",
        running=state["running"], status_text="running" if ready else state["status"],
        health="healthy" if ready else "starting" if state["running"] else "",
        port=str(state.get("port", profile.port)), gpu_id=profile.gpu_id,
        model=profile.model_path or profile.hf_repo or profile.model_file,
        pid=state["pid"], executable=state.get("executable", profile.server_binary),
        exit_code=state.get("exit_code"), ready=bool(ready))


_FLAG_CACHE: dict[tuple, set[str]] = {}
async def flags(profile) -> set[str]:
    import re
    env = environment(profile)
    binary = executable(profile, env)
    stat = Path(binary).stat()
    key = (binary, stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
    if key in _FLAG_CACHE:
        return set(_FLAG_CACHE[key])
    proc = await asyncio.create_subprocess_exec(binary, "--help", env=env,
        cwd=str(profile_store.PROJECT_ROOT), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), 15)
        if proc.returncode:
            raise RuntimeError(f"Host flag inspection failed (exit {proc.returncode})")
        result = set(re.findall(r"--([a-zA-Z0-9][a-zA-Z0-9-]*)", out.decode(errors="replace")))
        if not result:
            raise RuntimeError("No flags found in host executable --help")
        _FLAG_CACHE[key] = result
        return set(result)
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
