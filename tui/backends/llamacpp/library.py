"""Read-only catalog for local host llama.cpp assets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re

from tui.common import profile_store

KINDS = ("model", "mmproj", "draft", "template")


def format_size(size_bytes: int) -> str:
    """Format an asset size for the CLI and TUI catalog views."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size_bytes < 1024 or unit == "TiB":
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TiB"
_ROLE_DIRS = {
    "mmproj": "mmproj",
    "draft": "draft",
    "mtp": "draft",
    "chat-templates": "template",
}


@dataclass(frozen=True)
class Asset:
    path: str
    relative_path: str
    root: str
    kind: str
    size_bytes: int
    modified_at: float

    def as_dict(self) -> dict:
        return asdict(self)


def configured_roots() -> list[Path]:
    defaults = profile_store.effective_defaults("llamacpp")
    roots = defaults.get("library_roots", [])
    if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
        raise ValueError("defaults.llamacpp.library_roots must be a list of paths")
    return [Path(root).expanduser().absolute() for root in roots]


def _kind(relative: Path) -> str | None:
    suffix = relative.suffix.lower()
    if suffix == ".jinja":
        return "template"
    if suffix != ".gguf":
        return None
    for parent in relative.parents:
        role = _ROLE_DIRS.get(parent.name.lower())
        if role:
            return role
    return "model"


def scan(roots: list[Path] | None = None) -> tuple[list[Asset], list[str]]:
    """Return readable asset metadata and complete root/traversal errors.

    The scan deliberately does not open GGUF contents, hash files, or follow
    directory symlinks. File symlinks are retained because they are useful for
    Hugging Face snapshot-based stores.
    """
    assets: list[Asset] = []
    errors: list[str] = []
    for root in roots if roots is not None else configured_roots():
        try:
            root = root.resolve(strict=True)
            if not root.is_dir():
                raise NotADirectoryError(root)
            if not os.access(root, os.R_OK | os.X_OK):
                raise PermissionError(root)
        except OSError as exc:
            errors.append(f"{root}: {exc.strerror or type(exc).__name__}")
            continue
        try:
            for directory, directories, filenames in os.walk(root, followlinks=False):
                directories[:] = [
                    name for name in directories
                    if not (Path(directory) / name).is_symlink()
                ]
                for filename in filenames:
                    candidate = Path(directory) / filename
                    relative = candidate.relative_to(root)
                    kind = _kind(relative)
                    if kind is None:
                        continue
                    try:
                        stat = candidate.stat()
                        if not candidate.is_file() or not os.access(candidate, os.R_OK):
                            raise PermissionError(candidate)
                    except OSError as exc:
                        errors.append(f"{candidate}: {exc.strerror or type(exc).__name__}")
                        continue
                    assets.append(Asset(
                        path=str(candidate.absolute()), relative_path=relative.as_posix(),
                        root=str(root), kind=kind, size_bytes=stat.st_size,
                        modified_at=stat.st_mtime,
                    ))
        except OSError as exc:
            errors.append(f"{root}: {exc.strerror or type(exc).__name__}")
    return sorted(assets, key=lambda asset: (asset.kind, asset.relative_path.lower())), errors


def rank_companions(model: Asset, assets: list[Asset], kind: str) -> list[Asset]:
    """Suggest, but never select, assets whose names share model tokens."""
    if kind not in KINDS or model.kind != "model":
        return []
    tokens = {
        token for token in re.split(r"[^a-z0-9]+", Path(model.path).stem.lower())
        if len(token) > 2 and token not in {"gguf", "qwen", "model", "uncensored"}
    }
    def score(asset: Asset) -> tuple[int, str]:
        candidate = set(re.split(r"[^a-z0-9]+", Path(asset.path).stem.lower()))
        return (-len(tokens & candidate), asset.relative_path.lower())
    return sorted((asset for asset in assets if asset.kind == kind), key=score)
