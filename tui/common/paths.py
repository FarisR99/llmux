"""Authoritative separation of immutable llmux assets from mutable state."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class PathResolutionError(RuntimeError):
    pass


def _expand(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser().resolve()


def _is_code_root(path: Path) -> bool:
    return (path / "compose").is_dir() and (path / "profiles.example.yaml").is_file()


def _writable_ancestor(path: Path) -> Path:
    """Return the nearest existing ancestor, without creating state on import."""
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


@dataclass(frozen=True)
class LlmuxPaths:
    code_root: Path
    data_root: Path
    data_source: str

    @property
    def profiles_yaml(self) -> Path:
        return self.data_root / "profiles.yaml"

    @property
    def common_env(self) -> Path:
        return self.data_root / ".env.common"

    @property
    def config_dir(self) -> Path:
        return self.data_root / "config"

    @property
    def runtime_dir(self) -> Path:
        return self.data_root / ".runtime"


def resolve_paths() -> LlmuxPaths:
    root_value = os.environ.get("LLMUX_ROOT", "").strip()
    if root_value:
        code_root = _expand(root_value)
        if not _is_code_root(code_root):
            raise PathResolutionError(f"LLMUX_ROOT={root_value} is not an llmux checkout")
    else:
        cwd = Path.cwd().resolve()
        if _is_code_root(cwd):
            code_root = cwd
        else:
            package_root = Path(__file__).resolve().parents[1]
            code_root = package_root / "_assets"
            if not _is_code_root(code_root):
                # Editable/source installs retain assets beside the package.
                code_root = package_root.parent
            if not _is_code_root(code_root):
                raise PathResolutionError(f"llmux package assets are unavailable at {code_root}")

    data_value = os.environ.get("LLMUX_DATA_DIR", "").strip()
    if data_value:
        data_root, source = _expand(data_value), "LLMUX_DATA_DIR"
    elif root_value or _is_code_root(Path.cwd().resolve()):
        data_root, source = code_root, "legacy-checkout"
    else:
        data_root, source = _expand("~/.local/share/llmux"), "default"
    if data_root.exists() and not data_root.is_dir():
        raise PathResolutionError(f"data root is not a directory: {data_root}")
    parent = data_root if data_root.exists() else _writable_ancestor(data_root.parent)
    if not parent.exists() or not os.access(parent, os.W_OK | os.X_OK):
        raise PathResolutionError(f"data root is not writable: {data_root}")
    return LlmuxPaths(code_root, data_root, source)


PATHS = resolve_paths()
