"""Cross-runtime lifecycle serialization for CLI and TUI callers."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Iterator

from tui.common import profile_store

BUSY_EXIT_CODE = 75


class OperationInProgress(RuntimeError):
    def __init__(self, profile: str, requested_action: str, current: dict | None) -> None:
        self.profile = profile
        self.requested_action = requested_action
        self.current = current or {}
        super().__init__(f"Operation already in progress for {profile}")

    def payload(self) -> dict:
        return {
            "code": "OPERATION_IN_PROGRESS", "profile": self.profile,
            "requested_action": self.requested_action,
            "current_operation": self.current.get("action"),
            "current_state": self.current.get("state"),
            "retry_after_seconds": None,
            "retry_guidance": "Wait for the current operation to finish, then retry explicitly.",
        }


@contextmanager
def operation_lock(backend: str, profile: str, action: str, *, state: str = "running") -> Iterator[None]:
    """Acquire a non-blocking per-profile lock and expose safe owner metadata."""
    root = profile_store.RUNTIME_DIR / "lifecycle" / backend
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = root / f"{profile}.lock"
    metadata_path = root / f"{profile}.json"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            try:
                current = json.loads(metadata_path.read_text())
            except (OSError, ValueError):
                current = None
            raise OperationInProgress(profile, action, current) from exc
        profile_store._atomic_write(metadata_path, json.dumps({"action": action, "state": state, "started_at": time.time()}), mode=0o600)
        try:
            yield
        finally:
            metadata_path.unlink(missing_ok=True)
    finally:
        os.close(fd)
