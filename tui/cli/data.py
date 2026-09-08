"""Data-root diagnostics and explicit legacy-project migration."""
from __future__ import annotations

import hashlib
import shutil
import stat
from pathlib import Path

import typer

from tui.cli._runtime import emit_json
from tui.common import profile_store
from tui.common.paths import PATHS

app = typer.Typer(help="Mutable data-root diagnostics and migration.", no_args_is_help=True)


def _inventory(root: Path) -> list[Path]:
    names = [Path("profiles.yaml"), Path(".env.common"), Path("config")]
    return [root / name for name in names if (root / name).exists()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unsafe_permissions(path: Path) -> bool:
    return path.exists() and bool(stat.S_IMODE(path.stat().st_mode) & 0o077)


def _validate_declarative_state(root: Path) -> None:
    """Parse all declarative files without exposing their contents."""
    original = profile_store.PROFILES_YAML
    try:
        profile_store.PROFILES_YAML = root / "profiles.yaml"
        if profile_store.PROFILES_YAML.exists():
            profile_store.list_profiles("vllm")
            profile_store.list_profiles("llamacpp")
        from tui.common.config_markers import load_yaml_mapping
        for config in (root / "config").glob("*/*.yaml") if (root / "config").exists() else ():
            load_yaml_mapping(config.read_text(), config)
    finally:
        profile_store.PROFILES_YAML = original


def _legacy_source(value: str) -> Path:
    source = Path(value).expanduser().resolve() if value else PATHS.code_root
    if not (source / "compose").is_dir() or not (source / "profiles.example.yaml").is_file():
        raise typer.BadParameter("source must be an llmux checkout", param_hint="--source")
    return source


@app.command("status")
def status(json_out: bool = typer.Option(False, "--json")) -> None:
    """Show code/data roots and legacy migration readiness without secrets."""
    legacy = PATHS.code_root != PATHS.data_root and bool(_inventory(PATHS.code_root))
    result = {"code_root": str(PATHS.code_root), "data_root": str(PATHS.data_root),
              "data_source": PATHS.data_source, "legacy_state_detected": legacy,
              "migration_ready": legacy and not PATHS.data_root.exists(),
              "unsafe_permissions": {
                  "data_root": _unsafe_permissions(PATHS.data_root),
                  "env_common": _unsafe_permissions(PATHS.common_env),
              }}
    if json_out:
        emit_json({"status": "ok", "result": result})
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


@app.command("migrate")
def migrate(
    source: str = typer.Option("", "--source", help="Legacy checkout root (default: resolved code root)."),
    execute: bool = typer.Option(False, "--execute", help="Copy after validation; omit for a dry-run."),
    yes: bool = typer.Option(False, "--yes", help="Required with --execute for non-interactive migration."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Plan or copy legacy checkout state into the selected data root."""
    try:
        src = _legacy_source(source)
        dst = PATHS.data_root
        items = _inventory(src)
        if src == dst:
            raise RuntimeError("SOURCE_IS_DATA_ROOT: source already is the active data root")
        if not items:
            raise RuntimeError("LEGACY_STATE_NOT_FOUND: no mutable legacy state found")
        if dst.exists() and any(dst.iterdir()):
            raise RuntimeError("DESTINATION_NOT_EMPTY: choose an empty LLMUX_DATA_DIR")
        for state_path in (src / ".runtime").glob("llamacpp/host/*/state.json") if (src / ".runtime").exists() else ():
            try:
                import json
                state = json.loads(state_path.read_text())
            except (OSError, ValueError):
                raise RuntimeError("SOURCE_STATE_UNREADABLE: cannot safely inspect host runtime state") from None
            if state.get("status") in {"starting", "running", "stopping"}:
                raise RuntimeError("ACTIVE_PROFILES: stop managed profiles before migration")
        # Validate declarative state before creating a destination.
        _validate_declarative_state(src)
        manifest = [{"path": str(item.relative_to(src)), "files": sum(1 for p in item.rglob("*") if p.is_file()) if item.is_dir() else 1}
                    for item in items]
        result = {"source": str(src), "destination": str(dst), "items": manifest,
                  "copied": False, "not_copied": [".runtime", "locks", "PID/process state"],
                  "rollback": f"unset LLMUX_DATA_DIR (or set it to {src})"}
        if execute:
            if not yes:
                raise RuntimeError("CONFIRMATION_REQUIRED: pass --execute --yes")
            dst.mkdir(parents=True, mode=0o700)
            dst.chmod(0o700)
            for item in items:
                target = dst / item.name
                if item.is_dir():
                    shutil.copytree(item, target)
                    for child in target.rglob("*"):
                        child.chmod(0o600 if child.is_file() else 0o700)
                    target.chmod(0o700)
                else:
                    shutil.copy2(item, target)
                    target.chmod(0o600)
            for item in items:
                target = dst / item.name
                if item.is_file() and _digest(item) != _digest(target):
                    raise RuntimeError("COPY_VERIFICATION_FAILED: copied file differs from source")
                if item.is_dir():
                    source_files = sorted(p.relative_to(item) for p in item.rglob("*") if p.is_file())
                    target_files = sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file())
                    if source_files != target_files or any(_digest(item / name) != _digest(target / name) for name in source_files):
                        raise RuntimeError("COPY_VERIFICATION_FAILED: copied configuration differs from source")
            _validate_declarative_state(dst)
            result["copied"] = True
        if json_out:
            emit_json({"status": "ok", "result": result})
        else:
            print("Migration " + ("completed" if execute else "plan"))
            print(f"Source: {src}\nDestination: {dst}\nRollback: {result['rollback']}")
    except (OSError, RuntimeError, ValueError, typer.BadParameter) as exc:
        code, _, message = str(exc).partition(": ")
        if json_out:
            emit_json({"status": "error", "error": {"code": code if message else "MIGRATION_FAILED", "message": message or str(exc)}})
        else:
            typer.echo(f"Migration refused: {exc}", err=True)
        raise typer.Exit(code=1)
