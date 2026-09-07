"""Shared Docker and host llama-server argument rendering."""
from __future__ import annotations
import shlex
import sys

_MANAGED_FLAGS = frozenset(
    {"--host", "--port", "--webui", "--no-webui", "--ui", "--no-ui", "--metrics"}
)


def render_command(
    cfg: dict,
    *,
    model_file: str = "",
    hf_repo: str = "",
    hf_file: str = "",
    local_model: str = "",
    mmproj: str = "",
    draft_model: str = "",
    draft_type: str = "",
    mtp_n: int = 0,
    chat_template: str = "",
    host: str = "0.0.0.0",
    port: int = 8080,
) -> list[str]:
    cfg = dict(cfg)
    args: list[str] = [
        "--host", host, "--port", str(port), "--no-webui", "--metrics",
    ]

    configured_model_file = cfg.pop("model-file", "")
    if configured_model_file is not None and not isinstance(configured_model_file, str):
        raise ValueError(
            "config key 'model-file' must be a string; "
            f"got {type(configured_model_file).__name__}"
        )
    resolved_file = (hf_file or configured_model_file or model_file).strip()
    resolved_repo = hf_repo.strip()
    if not local_model and not resolved_repo:
        raise ValueError(
            "hf_repo is empty — set profile.hf_repo so llama-server can `-hf` download"
        )
    if not local_model and not resolved_file:
        raise ValueError("hf_file / model_file is empty — need a specific .gguf filename")
    if local_model:
        args.extend(["--model", local_model])
    else:
        args.extend(["-hf", resolved_repo])
        args.extend(["-hff", resolved_file])
    cfg.pop("model-file", None)

    host_assets = {
        "mmproj": mmproj,
        "spec-draft-model": draft_model,
        "spec-type": draft_type,
        "spec-draft-n-max": mtp_n,
        "chat-template-file": chat_template,
    }
    if local_model:
        for key, value in host_assets.items():
            if key in cfg:
                raise ValueError(f"config cannot override host bundle asset: {key}")
            if value:
                args.extend([f"--{key}", str(value)])

    for key in ("no-webui", "webui", "ui", "no-ui", "metrics", "host", "port"):
        if key in cfg:
            print(
                f"warning: config key '{key}' is managed by llmux and was ignored",
                file=sys.stderr,
            )
            cfg.pop(key, None)

    override_tensors = _string_list(
        cfg.pop("override-tensors", None), "override-tensors"
    )
    extra_args_value = cfg.pop("extra-args", None)
    if isinstance(extra_args_value, str):
        extra_args = shlex.split(extra_args_value)
    else:
        extra_args = _string_list(extra_args_value, "extra-args")
    _validate_extra_args(extra_args)
    if local_model:
        managed = {
            "-m", "--model", "-hf", "--hf-repo", "-hff", "--hf-file",
            "--model-url", "-mu", "--hf-repo-v", "-hfr", "--hf-repo-draft",
            "-hfd", "-mm", "--mmproj", "-md", "--spec-draft-model",
            "--spec-type", "--spec-draft-n-max", "--chat-template-file",
        }
        for token in extra_args:
            if token.split("=", 1)[0] in managed:
                raise ValueError(f"extra-args cannot override model selection: {token}")
        for key in cfg:
            if f"--{key}" in managed:
                raise ValueError(f"config cannot override model selection: {key}")

    _BOOL_VALUE_FLAGS = {"flash-attn"}

    for key, value in cfg.items():
        if not isinstance(key, str):
            raise ValueError(
                f"config keys must be strings; got {type(key).__name__}"
            )
        flag = f"--{key}"
        if isinstance(value, bool):
            if key in _BOOL_VALUE_FLAGS:
                args.extend([flag, "on" if value else "off"])
            elif value:
                args.append(flag)
        elif isinstance(value, list):
            for item in value:
                args.extend([flag, _scalar_text(item, key)])
        elif value is None:
            continue
        else:
            args.extend([flag, _scalar_text(value, key)])

    for pattern in override_tensors:
        args.extend(["-ot", str(pattern)])

    args.extend(str(a) for a in extra_args)
    return args


def _scalar_text(value: object, key: str) -> str:
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(
            f"config key '{key}' must be a scalar or list of scalar values; "
            f"got {type(value).__name__}"
        )
    return str(value)


def _string_list(value: object, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a string or a list of strings")
    return value


def _validate_extra_args(extra_args: list[str]) -> None:
    for token in extra_args:
        option = token.split("=", 1)[0]
        if option in _MANAGED_FLAGS:
            raise ValueError(f"extra-args cannot override llmux-managed option {option}")
