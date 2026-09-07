"""Linux process ownership and detached supervision. No Docker dependency."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from typing import Iterator

from tui.common import profile_store


def require_linux() -> None:
    if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Host runtime requires Linux with pidfd support (kernel 5.3+)")


def directory(name: str) -> Path:
    if not profile_store._NAME_RE.fullmatch(name):
        raise ValueError("invalid host profile name")
    return profile_store.RUNTIME_DIR / "llamacpp" / "host" / name


def ensure_directory(name: str) -> Path:
    require_linux()
    path = directory(name)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def identity(pid: int) -> dict | None:
    """Read birth identity, distinguishing absence from inaccessible /proc."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start": fields[19],
                "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record: dict | None) -> bool:
    if not record:
        return False
    if not isinstance(record, dict) or type(record.get("pid")) is not int or record["pid"] <= 0:
        raise RuntimeError("Invalid host process identity")
    return identity(record["pid"]) == record


def send_signal(record: dict | None, sig: int) -> bool:
    require_linux()
    if not record:
        return False
    # Open first, then verify birth identity. The fd cannot retarget a reused PID.
    pid = record.get("pid")
    if type(pid) is not int or pid <= 0:
        raise RuntimeError("Invalid host process identity")
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        if not alive(record):
            return False
        signal.pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def read_state(name: str) -> dict:
    try:
        value = json.loads((directory(name) / "state.json").read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read host state for {name}: {exc}") from exc
    if not isinstance(value, dict) or value.get("version") != 1:
        raise RuntimeError(f"Invalid host state for {name}")
    return value


def write_state(name: str, state: dict) -> None:
    profile_store._atomic_write(directory(name) / "state.json", json.dumps(state), mode=0o600)


def status(name: str) -> dict:
    state = read_state(name)
    server_alive = alive(state.get("server"))
    supervisor_alive = alive(state.get("supervisor"))
    active = server_alive or supervisor_alive
    phase = state.get("status", "stopped")
    if active and not supervisor_alive:
        phase = "orphaned"
    elif not active and phase not in ("stopped", "exited", "failed"):
        phase = "exited"
    return {**state, "active": active, "running": server_alive,
            "status": phase, "pid": (state.get("server") or {}).get("pid") if server_alive else None}


@contextmanager
def lifecycle_lock(name: str) -> Iterator[None]:
    path = ensure_directory(name) / "lifecycle.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Host lifecycle operation already in progress for {name}") from exc
        yield
    finally:
        os.close(fd)


def require_stopped(name: str) -> None:
    # Checking the lock also excludes the short launch interval before state exists.
    with lifecycle_lock(name):
        if status(name)["active"]:
            raise ValueError(f"Host profile '{name}' is active; stop it first")


async def logs(name: str, *, tail: int = 100, follow: bool = True):
    from collections import deque
    if tail < 0:
        raise ValueError("tail must be non-negative")
    path = directory(name) / "server.log"
    if not path.exists():
        if not read_state(name):
            raise RuntimeError(f"No host logs for '{name}'; it has not been started")
        return
    recent: deque[str] = deque(maxlen=tail)
    if tail:
        for index in (3, 2, 1):
            rotated = path.with_name(f"server.log.{index}")
            if rotated.exists():
                with rotated.open(errors="replace") as old:
                    recent.extend(old)
    stream = path.open(errors="replace")
    try:
        recent.extend(stream)
        for line in recent:
            yield line.rstrip("\n")
        inode = os.fstat(stream.fileno()).st_ino
        while follow:
            line = stream.readline()
            if line:
                yield line.rstrip("\n")
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                await asyncio.sleep(0.1)
                continue
            if stat.st_ino != inode:
                stream.close()
                stream = path.open(errors="replace")
                inode = os.fstat(stream.fileno()).st_ino
                continue
            if not status(name)["active"]:
                return
            await asyncio.sleep(0.2)
    finally:
        stream.close()


def supervise(name: str) -> int:
    """Internal worker: receives argv/env via stdin, never persists secrets."""
    require_linux()
    os.umask(0o077)
    spec = json.load(sys.stdin)
    path = ensure_directory(name)
    logger = logging.getLogger("llmux.host")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(path / "server.log", maxBytes=10 * 1024 * 1024,
                                  backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    state = {"version": 1, "supervisor": identity(os.getpid()), "status": "starting",
             "executable": spec["argv"][0], "port": spec["port"], "exit_code": None,
             "started_at": time.time(),
             "command_hash": __import__("hashlib").sha256(json.dumps(spec["argv"]).encode()).hexdigest()}
    write_state(name, state)
    stopping: list[float] = []
    def request_stop(signum, frame):
        if not stopping:
            stopping.append(time.monotonic())
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    proc = None
    try:
        proc = subprocess.Popen(spec["argv"], env=spec["env"], cwd=spec["cwd"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        state["server"] = identity(proc.pid)
        write_state(name, state)
        os.set_blocking(proc.stdout.fileno(), False)
        buffer = b""
        sent_term = False
        while True:
            if stopping and not sent_term:
                state["status"] = "stopping"
                write_state(name, state)
                send_signal(state.get("server"), signal.SIGTERM)
                sent_term = True
            if stopping and time.monotonic() - stopping[0] >= 30:
                send_signal(state.get("server"), signal.SIGKILL)
            ready, _, _ = select.select([proc.stdout], [], [], 0.2)
            if ready:
                chunk = os.read(proc.stdout.fileno(), 65536)
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    logger.info(line.decode(errors="replace"))
                if len(buffer) >= 65536:
                    logger.info(buffer.decode(errors="replace"))
                    buffer = b""
                if not chunk and proc.poll() is not None:
                    break
            if proc.poll() is not None and not ready:
                break
        if buffer:
            logger.info(buffer.decode(errors="replace"))
        state["exit_code"] = proc.wait()
        state["status"] = "stopped" if stopping else "exited"
    except Exception as exc:
        logger.exception("Host supervisor failed")
        if proc is not None:
            proc.kill()
            proc.wait()
        state.update(status="failed", error=str(exc))
    finally:
        write_state(name, state)
        handler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(supervise(sys.argv[1]))
