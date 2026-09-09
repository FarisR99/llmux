"""Hermetic tests for exact local tokenizer adapters."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tui.common.tokens import (
    TokenizerOutputError,
    TokenizerProcessError,
    count_llamacpp,
)


def _result(argv: list[str], rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, rc, stdout, stderr)


@pytest.fixture
def tokenizer(tmp_path: Path) -> str:
    binary = tmp_path / "llama-tokenize"
    binary.write_text("placeholder")
    binary.chmod(0o700)
    return str(binary)


def test_llamacpp_show_count_uses_stdin_and_parses_informational_output(tokenizer: str) -> None:
    prompt = "do not put this in argv"
    with patch("tui.common.tokens.subprocess.run") as run:
        run.return_value = _result([], 0, "load time = 1ms\ntoken count: 42\n")
        assert count_llamacpp(tokenizer, "model.gguf", prompt) == (42, "llamacpp_show_count")
    argv = run.call_args.args[0]
    assert argv == [tokenizer, "--model", "model.gguf", "--stdin", "--show-count"]
    assert prompt not in argv
    assert run.call_args.kwargs["input"] == prompt.encode("utf-8")


def test_llamacpp_legacy_json_fallback_is_stdin_only(tokenizer: str) -> None:
    prompt = "private input"
    standard = _result([], 1, stderr="unknown option: --show-count")
    legacy = _result([], 0, '{"tokens": 17}')
    with patch("tui.common.tokens.subprocess.run", side_effect=[standard, legacy]) as run:
        assert count_llamacpp(tokenizer, "model.gguf", prompt) == (17, "llamacpp_json")
    assert run.call_args_list[1].args[0] == [tokenizer, "--model", "model.gguf", "--stdin", "--json"]
    assert all(prompt not in call.args[0] for call in run.call_args_list)
    assert all(call.kwargs["input"] == prompt.encode("utf-8") for call in run.call_args_list)


@pytest.mark.parametrize("output", ["token count: 4\ntokens: 5\n", "token count: nope\n", ""])
def test_llamacpp_rejects_ambiguous_or_invalid_count_output(tokenizer: str, output: str) -> None:
    with patch("tui.common.tokens.subprocess.run", return_value=_result([], 0, output)):
        with pytest.raises(TokenizerOutputError):
            count_llamacpp(tokenizer, "model.gguf", "input")


def test_llamacpp_process_failure_includes_safe_bounded_diagnostic(tokenizer: str) -> None:
    prompt = "never expose this"
    failure = _result([], 1, stderr=f"CUDA driver mismatch while processing {prompt}")
    with patch("tui.common.tokens.subprocess.run", return_value=failure):
        with pytest.raises(TokenizerProcessError) as exc:
            count_llamacpp(tokenizer, "model.gguf", prompt)
    assert "CUDA driver mismatch" in str(exc.value)
    assert prompt not in str(exc.value)


def test_llamacpp_never_estimates_after_bad_standard_output(tokenizer: str) -> None:
    with patch("tui.common.tokens.subprocess.run", return_value=_result([], 0, "tokens: 4\ntokens: 5")) as run:
        with pytest.raises(TokenizerOutputError):
            count_llamacpp(tokenizer, "model.gguf", "input")
    assert run.call_count == 1


def test_llamacpp_decodes_invalid_utf8_before_parsing_exact_count(tokenizer: str) -> None:
    result = _result([], 0, b"model diagnostic: \xff\xfe\ntoken count: 42\n")
    with patch("tui.common.tokens.subprocess.run", return_value=result):
        assert count_llamacpp(tokenizer, "model.gguf", "input") == (42, "llamacpp_show_count")


def test_llamacpp_decodes_invalid_utf8_in_failure_diagnostic(tokenizer: str) -> None:
    failure = _result([], 1, b"", b"CUDA runtime \xff incompatible")
    with patch("tui.common.tokens.subprocess.run", return_value=failure):
        with pytest.raises(TokenizerProcessError, match="CUDA runtime"):
            count_llamacpp(tokenizer, "model.gguf", "input")
