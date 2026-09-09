"""Exact local token-count adapters. Never fall back to estimates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess


class TokenCapabilityError(RuntimeError):
    code = "TOKENIZER_UNAVAILABLE"


class TokenInputError(ValueError):
    code = "INPUT_INVALID"


class TokenizerContractError(TokenCapabilityError):
    """The selected binary does not expose a safe, supported interface."""
    code = "TOKENIZER_UNSUPPORTED"


class TokenizerProcessError(TokenCapabilityError):
    """The tokenizer accepted the request but could not run it."""
    code = "TOKENIZER_FAILED"


class TokenizerOutputError(TokenCapabilityError):
    """The tokenizer completed but did not return one exact count."""
    code = "TOKENIZER_OUTPUT_INVALID"


def count_vllm(model: str, text: str, *, messages: list[dict] | None = None) -> tuple[int, str, int | None]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenCapabilityError("local transformers support is not installed") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    except Exception as exc:
        raise TokenCapabilityError("local tokenizer assets are unavailable") from exc
    try:
        if messages is None:
            count = len(tokenizer.encode(text, add_special_tokens=True))
            mode = "raw_text"
        else:
            if not hasattr(tokenizer, "apply_chat_template") or not tokenizer.chat_template:
                raise TokenCapabilityError("the local tokenizer has no chat template")
            count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
            mode = "chat_messages"
    except TokenCapabilityError:
        raise
    except Exception as exc:
        raise TokenInputError("messages cannot be rendered by the local chat template") from exc
    limit = getattr(tokenizer, "model_max_length", None)
    if not isinstance(limit, int) or limit <= 0 or limit > 10**9:
        limit = None
    return count, mode, limit


def _safe_stderr_summary(stderr: str, text: str) -> str:
    """Return a bounded diagnostic without allowing a reflected prompt through."""
    summary = " ".join(stderr.split())
    if text and text in summary:
        summary = summary.replace(text, "[redacted prompt]")
    return summary[:300]


def _run_tokenizer(argv: list[str], text: str) -> subprocess.CompletedProcess[str]:
    """Run a tokenizer with UTF-8 stdin and loss-tolerant diagnostic decoding."""
    result = subprocess.run(
        argv, input=text.encode("utf-8"), capture_output=True, timeout=30, check=False
    )

    def decode(value: bytes | str | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or ""

    # llama.cpp diagnostics may contain model bytes that are not valid UTF-8.
    # Decode only after process completion so they cannot bypass exact parsing.
    return subprocess.CompletedProcess(
        result.args, result.returncode, decode(result.stdout), decode(result.stderr)
    )


def _parse_show_count(output: str) -> int:
    """Parse exactly one labelled count from conventional llama-tokenize output."""
    patterns = (
        r"\b(?:token\s+count|tokens?|n_tokens)\s*[:=]\s*(\d+)\b",
        r"^\s*(\d+)\s+tokens?\s*$",
    )
    matches: list[str] = []
    for line in output.splitlines():
        for pattern in patterns:
            matches.extend(re.findall(pattern, line, flags=re.IGNORECASE))
    if len(matches) != 1:
        raise TokenizerOutputError(
            "local llama tokenizer output did not contain one unambiguous token count"
        )
    try:
        count = int(matches[0])
    except ValueError as exc:  # Kept defensive if regexes change.
        raise TokenizerOutputError("local llama tokenizer returned a malformed token count") from exc
    if count < 0:
        raise TokenizerOutputError("local llama tokenizer returned a negative token count")
    return count


def _option_error(result: subprocess.CompletedProcess[str]) -> bool:
    """Whether a failed process rejected its CLI flags before loading a model."""
    message = f"{result.stdout}\n{result.stderr}".lower()
    return any(marker in message for marker in (
        "unknown option", "unknown argument", "unrecognized option", "unsupported option",
    ))


def count_llamacpp(binary: str, model: str, text: str) -> tuple[int, str]:
    """Count GGUF tokens exactly using a stdin-only llama-tokenize interface.

    Current llama.cpp uses ``--stdin --show-count``.  A second, stdin-capable
    JSON invocation keeps compatibility with custom tokenizer wrappers, but is
    attempted only when the standard flags are explicitly rejected.
    """
    path = Path(binary).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise TokenCapabilityError("a local llama-tokenize binary is required for exact GGUF tokenization")
    try:
        result = _run_tokenizer([str(path), "--model", model, "--stdin", "--show-count"], text)
    except OSError as exc:
        raise TokenCapabilityError("local llama-tokenize binary is missing or is not executable") from exc
    if not result.returncode:
        return _parse_show_count(result.stdout), "llamacpp_show_count"

    # Old custom wrappers may expose JSON, but prompt text must still stay on
    # stdin.  Do not retry runtime/model failures: that needlessly reloads a
    # model and conceals useful diagnostics such as CUDA incompatibilities.
    if not _option_error(result):
        detail = _safe_stderr_summary(result.stderr, text)
        suffix = f": {detail}" if detail else ""
        raise TokenizerProcessError(f"local llama tokenizer failed while counting{suffix}")
    try:
        legacy = _run_tokenizer([str(path), "--model", model, "--stdin", "--json"], text)
    except OSError as exc:
        raise TokenCapabilityError("local llama-tokenize binary is missing or is not executable") from exc
    if legacy.returncode:
        if _option_error(legacy):
            raise TokenizerContractError(
                "local llama-tokenize must support --stdin --show-count or --stdin --json"
            )
        detail = _safe_stderr_summary(legacy.stderr, text)
        suffix = f": {detail}" if detail else ""
        raise TokenizerProcessError(f"local llama tokenizer failed while counting{suffix}")
    try:
        data = json.loads(legacy.stdout)
        count = data.get("tokens") if isinstance(data, dict) else None
    except ValueError as exc:
        raise TokenizerOutputError("local llama tokenizer did not return a JSON token count") from exc
    if type(count) is not int or count < 0:
        raise TokenizerOutputError("local llama tokenizer returned an invalid token count")
    return count, "llamacpp_json"


def context_limit(backend: str, params: dict) -> int | None:
    key = "max-model-len" if backend == "vllm" else "ctx-size"
    value = params.get(key)
    return value if type(value) is int and value > 0 else None
