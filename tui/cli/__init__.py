"""llmux CLI — headless interface mirroring TUI features.

The TUI is still the default: `llmux` with no arguments launches it.
Subcommands provide non-interactive access for scripts and agents.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer

app = typer.Typer(
    name="llmux",
    help="Terminal multiplexer for vLLM + llama.cpp. "
    "Run with no arguments to launch the TUI; use subcommands for headless control.",
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _interactive_session() -> bool:
    """True only for a genuine interactive terminal session.

    Returns False under CI, or when `LLMUX_NONINTERACTIVE` is set, even if a
    pseudo-TTY is allocated — so first-run onboarding and the update prompt
    never block an automated job that happens to run with a TTY attached
    (`docker run -t`, `ssh -t`, some CI runners).
    """
    import os
    import sys

    if os.environ.get("CI") or os.environ.get("LLMUX_NONINTERACTIVE"):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def _maybe_run_onboarding() -> None:
    """Run the first-run wizard once, when `.env.common` is missing and the
    session is interactive.

    Headless invocations (scripts, pipes, CI) are left untouched so they fall
    back to the normal validation error instead of blocking on a prompt.
    Onboarding is best-effort — any failure is swallowed so it never breaks
    startup.
    """
    try:
        from tui.common.onboarding import needs_onboarding, run_onboarding

        if not needs_onboarding():
            return
        if not _interactive_session():
            return
        run_onboarding()
    except Exception as exc:  # noqa: BLE001 — never fatal, but never silent
        import sys

        print(f"Warning: onboarding did not complete ({exc}).", file=sys.stderr)


def _maybe_check_for_update() -> None:
    """Check for a newer llmux release, in interactive sessions.

    Runs on every interactive invocation (only a failed lookup backs off), so
    a release published minutes ago shows up on the next command. Headless
    invocations skip it so scripts and CI never see a prompt or extra output.
    SystemExit (raised after a successful self-update so the user restarts on
    fresh code) is allowed to propagate; everything else is swallowed — a
    version check must never break startup.
    """
    if not _interactive_session():
        return
    try:
        from tui.common.version_check import check_for_update

        check_for_update()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — never fatal, but never silent
        import sys

        print(f"Warning: update check failed ({exc}).", file=sys.stderr)


def llmux_version() -> str:
    """Return installed metadata or the source checkout's declared version."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("llmux")
    except PackageNotFoundError:
        from tui.common.profile_store import CODE_ROOT

        pyproject = CODE_ROOT / "pyproject.toml"
        try:
            content = pyproject.read_text()
        except OSError as exc:
            raise RuntimeError(
                f"llmux package metadata is unavailable and {pyproject} could not be read: {exc}"
            ) from exc
        project = re.search(
            r"^\[project\]\s*$\n(?P<body>.*?)(?=^\[|\Z)",
            content,
            re.MULTILINE | re.DOTALL,
        )
        match = (
            re.search(r'^version\s*=\s*"([^"]+)"', project.group("body"), re.MULTILINE)
            if project
            else None
        )
        if match is None:
            raise RuntimeError(
                f"llmux package metadata is unavailable and [project].version is missing from {pyproject}"
            )
        return match.group(1)


def _version_callback(value: bool) -> None:
    if not value:
        return
    print(f"llmux {llmux_version()}")
    raise typer.Exit()


def _configure_logging(level: str) -> None:
    """Attach a stderr handler so the modules' log calls are actually visible.

    Without this the root logger has no handler: DEBUG records are dropped
    entirely and WARNING records only reach stderr via logging.lastResort,
    which the TUI swallows.
    """
    import logging

    resolved = getattr(logging, level.upper(), None)
    if not isinstance(resolved, int):
        raise typer.BadParameter(
            f"unknown log level {level!r} (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            param_hint="--log-level",
        )
    import sys

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    # Scoped to llmux's own logger: a root handler at DEBUG would drown the
    # output in httpcore/urllib3 chatter.
    logger = logging.getLogger("tui")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(resolved)
    logger.propagate = False


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show the llmux version and exit.",
    ),
    log_level: str = typer.Option(
        "WARNING", "--log-level",
        envvar="LLMUX_LOG_LEVEL",
        help="Log verbosity on stderr (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    ),
) -> None:
    """Default behavior: launch the TUI when no subcommand is given."""
    _configure_logging(log_level)
    if ctx.invoked_subcommand == "update":
        return
    else:
        _maybe_run_onboarding()
        _maybe_check_for_update()
    if ctx.invoked_subcommand is not None:
        return
    from tui.cli._launch_tui import launch_tui

    launch_tui()


@app.command("tui")
def tui_cmd() -> None:
    """Launch the interactive TUI explicitly."""
    from tui.cli._launch_tui import launch_tui

    launch_tui()


@app.command("top")
def top_cmd(
    profile: str = typer.Argument(
        None, help="Show only this running profile (default: every running model)."
    ),
) -> None:
    """Live btop-style system monitor in a plain terminal (no TUI).

    Always shows every GPU — util, memory, temperature, power, PCIe — whether or
    not anything is running, plus a panel per running model (throughput and KV
    graphs, cache hit, requests, latency percentiles). q or Ctrl+C to quit."""
    from tui.common.plain_monitor import run_cli

    raise typer.Exit(code=run_cli(profile))


@app.command("update")
def update_cmd(
    check: bool = typer.Option(
        False, "--check", help="Only report; never touch the checkout."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Update without asking for confirmation."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit the status as JSON instead of a human line."
    ),
) -> None:
    """Check for a newer llmux release right now, and update to it.

    Unlike the startup check this ignores the failure back-off and runs even
    in a non-interactive session, so it always answers. Exits non-zero when
    the check could not be completed or the update failed.
    """
    from tui.cli._runtime import emit_json
    from tui.common import version_check as vc

    status = vc.resolve_status(respect_cooldown=False)
    payload = {
        "state": status.state,
        "latest_tag": status.tag,
        "local_version": status.local_version,
        "url": status.url,
        "detail": status.detail,
        "checkout": str(vc.PROJECT_ROOT),
    }

    if status.state == vc.UNKNOWN:
        if json_out:
            emit_json(payload)
        else:
            typer.echo(f"could not check for updates — {status.detail}", err=True)
        raise typer.Exit(code=1)
    if status.state == vc.CURRENT:
        if json_out:
            emit_json(payload)
        else:
            print(f"llmux is up to date ({status.tag or status.local_version}).")
        raise typer.Exit(code=0)
    if check:
        if json_out:
            emit_json(payload)
        else:
            print(f"llmux {status.tag} is available.  {status.url}".rstrip())
            print(f"  local: {status.local_version or 'unknown'}  ({vc.PROJECT_ROOT})")
        raise typer.Exit(code=0)

    if not json_out:
        print(f"llmux {status.tag} is available.  {status.url}".rstrip())

    blocked = vc.update_blocked_reason()
    if blocked:
        message = (
            f"refusing to update automatically — {blocked}. "
            f"Pull it yourself in {vc.PROJECT_ROOT} when ready."
        )
        if json_out:
            payload["error"] = message
            emit_json(payload)
        else:
            typer.echo(message, err=True)
        raise typer.Exit(code=1)

    if json_out and not yes:
        payload["error"] = "--json update requires --yes or --check"
        emit_json(payload)
        raise typer.Exit(code=2)
    if not yes and not typer.confirm("Update now?", default=True):
        raise typer.Exit(code=0)

    ok, message = vc.apply_update(status.tag)
    if json_out:
        payload["update_ok"] = ok
        payload["message"] = message
        emit_json(payload)
        if not ok:
            raise typer.Exit(code=1)
        return
    if not ok:
        typer.echo(message, err=True)
        raise typer.Exit(code=1)
    print(message)


from tui.cli import config as _config  # noqa: E402
from tui.cli import data as _data  # noqa: E402
from tui.cli import container as _container  # noqa: E402
from tui.cli import image as _image  # noqa: E402
from tui.cli import library as _library  # noqa: E402
from tui.cli import profile as _profile  # noqa: E402
from tui.cli import system as _system  # noqa: E402

app.add_typer(_container.app, name="container", help="Container lifecycle commands.")
app.add_typer(_profile.app, name="profile", help="Profile CRUD + quick-setup.")
app.add_typer(_config.app, name="config", help="Config (YAML) CRUD.")
app.add_typer(_data.app, name="data", help="Mutable data-root diagnostics and migration.")
app.add_typer(_image.app, name="image", help="Docker image inventory + dev build.")
app.add_typer(_library.app, name="library", help="Local host llama.cpp asset catalog.")
app.add_typer(_system.app, name="system", help="System info & env validation.")
# Top-level shortcuts for the most-used system commands.
app.command("gpu", help="Print nvidia-smi summary (alias for `system gpu`).")(_system.gpu)
app.command("env-check", help="Validate .env.common (alias for `system env-check`).")(
    _system.env_check
)
# Top-level shortcuts so agents can write `llmux up <profile>` directly.
app.command("up", help="Start a profile's server (alias for `container up`).")(
    _container.up
)
app.command(
    "prepare",
    help="Download the model + render runtime files without starting "
    "(alias for `container prepare`).",
)(_container.prepare)
app.command("down", help="Stop a profile's server (alias for `container down`).")(
    _container.down
)
app.command("logs", help="Stream server logs (alias for `container logs`).")(
    _container.logs
)
app.command("ps", help="List profiles and their runtime status (alias for `container ps`).")(
    _container.ps
)
app.command(
    "render-env",
    help="Re-render runtime env file from profiles.yaml (alias for `container render-env`).",
)(_container.render_env)
app.command(
    "bench",
    help="Benchmark a running server's tok/s (alias for `container benchmark`).",
)(_container.benchmark)
app.command(
    "stats",
    help="Live tok/s from running servers' /metrics (alias for `container stats`).",
)(_container.stats)


@app.command("tokens")
def tokens_cmd(
    profile: str = typer.Argument(..., help="Profile whose local tokenizer is used."),
    stdin: bool = typer.Option(False, "--stdin", help="Read UTF-8 text or messages from stdin."),
    file: str = typer.Option("", "--file", help="Read UTF-8 text or messages from this file."),
    messages: bool = typer.Option(False, "--messages", help="Interpret input as a JSON message array."),
    reserve: int = typer.Option(0, "--reserve", min=0, help="Reserved output-token capacity."),
    backend: str | None = typer.Option(None, "--backend", "-b"),
    tokenizer_binary: str = typer.Option("", "--tokenizer-binary", help="Local llama-tokenize binary for GGUF profiles."),
    json_out: bool = typer.Option(False, "--json", help="Emit a structured token-count result."),
) -> None:
    """Count tokens without starting a server; never logs prompt contents."""
    import sys
    from tui.cli._runtime import detect_backend, emit_json
    from tui.common import profile_store
    from tui.common.tokens import TokenCapabilityError, TokenInputError, context_limit, count_llamacpp, count_vllm
    if stdin == bool(file):
        raise typer.BadParameter("choose exactly one of --stdin or --file")
    try:
        text = sys.stdin.read() if stdin else Path(file).read_text(encoding="utf-8")
        payload = json.loads(text) if messages else None
        if messages and (not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload)):
            raise TokenInputError("--messages input must be a JSON array of objects")
        bk = detect_backend(profile, override=backend)
        stored = profile_store.load_profile(profile, bk)
        if bk == "vllm":
            from tui.backends.vllm.backend_storage import load_config
            config = load_config(stored.config_name or stored.name)
            source = str(config.extra_params.get("tokenizer") or config.model or stored.model_id)
            count, mode, inferred_limit = count_vllm(source, text, messages=payload)
            limit = context_limit(bk, {"max-model-len": config.extra_params.get("max-model-len")}) or inferred_limit
        else:
            if messages:
                raise TokenCapabilityError("offline GGUF chat-message tokenization is unavailable without a template-capable tokenizer")
            from tui.backends.llamacpp.backend import load_config
            config = load_config(stored.config_name or stored.name)
            source = tokenizer_binary or ""
            model = stored.model_path or stored.hf_file or stored.model_file
            count, mode = count_llamacpp(source, model, text)
            limit = context_limit(bk, config.params)
        remaining = max(limit - count - reserve, 0) if limit is not None else None
        result = {"profile": profile, "backend": bk, "tokenizer_mode": mode,
                  "tokenizer_source": source, "input_tokens": count, "context_limit": limit,
                  "reserved_tokens": reserve, "remaining_capacity": remaining,
                  "fits_context": None if limit is None else count + reserve <= limit}
        if json_out:
            emit_json({"status": "ok", "result": result})
        else:
            print(f"{profile}: {count} input tokens")
    except TokenInputError as exc:
        if json_out:
            emit_json({"status": "error", "error": {"code": "INPUT_INVALID", "message": str(exc)}})
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2)
    except TokenCapabilityError as exc:
        if json_out:
            emit_json({"status": "error", "error": {"code": exc.code, "message": str(exc)}})
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entrypoint (`llmux` in pyproject.scripts)."""
    try:
        app()
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
