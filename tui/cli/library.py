"""Read-only local model-library inventory."""
from __future__ import annotations

from pathlib import Path

import typer

from tui.cli._runtime import emit_json, emit_table
from tui.backends.llamacpp import library

app = typer.Typer(help="Local host llama.cpp asset catalog.", no_args_is_help=True)


@app.command("list")
def list_assets(
    root: list[Path] = typer.Option([], "--root", help="Additional or replacement catalog root."),
    kind: str = typer.Option("", "--kind", help="model, mmproj, draft, or template."),
    json_out: bool = typer.Option(False, "--json", help="Emit catalog records as JSON."),
) -> None:
    """List local GGUF and Jinja assets without modifying them."""
    if kind and kind not in library.KINDS:
        raise typer.BadParameter("kind must be model, mmproj, draft, or template")
    assets, errors = library.scan(root or None)
    if kind:
        assets = [asset for asset in assets if asset.kind == kind]
    rows = [asset.as_dict() for asset in assets]
    if json_out:
        emit_json({"assets": rows, "errors": errors})
    else:
        emit_table(
            [
                {
                    "kind": asset.kind,
                    "path": asset.path,
                    "size_mib": f"{asset.size_bytes / 1024**2:.1f}",
                }
                for asset in assets
            ],
            columns=["kind", "size_mib", "path"],
        )
        for error in errors:
            typer.echo(f"catalog error: {error}", err=True)
    if errors:
        raise typer.Exit(code=1)
