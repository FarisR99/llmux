"""Exact local token-count adapters. Never fall back to estimates."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess


class TokenCapabilityError(RuntimeError):
    code = "TOKENIZER_UNAVAILABLE"


class TokenInputError(ValueError):
    code = "INPUT_INVALID"


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


def count_llamacpp(binary: str, model: str, text: str) -> tuple[int, str]:
    """Use a caller-selected llama-tokenize binary that emits `{tokens: N}` JSON."""
    path = Path(binary).expanduser()
    if not path.is_file() or not path.exists():
        raise TokenCapabilityError("a local llama-tokenize binary is required for exact GGUF tokenization")
    try:
        result = subprocess.run([str(path), "--model", model, "--prompt", text, "--json"],
                                capture_output=True, text=True, timeout=30, check=False)
    except OSError as exc:
        raise TokenCapabilityError("could not run the local llama tokenizer") from exc
    if result.returncode:
        raise TokenCapabilityError("local llama tokenizer could not count this model")
    try:
        data = json.loads(result.stdout)
        count = data.get("tokens")
    except (ValueError, AttributeError) as exc:
        raise TokenCapabilityError("local llama tokenizer did not return a JSON token count") from exc
    if type(count) is not int or count < 0:
        raise TokenCapabilityError("local llama tokenizer returned an invalid token count")
    return count, "raw_text"


def context_limit(backend: str, params: dict) -> int | None:
    key = "max-model-len" if backend == "vllm" else "ctx-size"
    value = params.get(key)
    return value if type(value) is int and value > 0 else None
