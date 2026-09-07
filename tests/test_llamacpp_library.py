from __future__ import annotations

from pathlib import Path

import pytest

from tui.backends.llamacpp import library
from tui.backends.llamacpp.command import render_command
from tui.common.profile_store import StoredProfile, _validate_profile


def test_scan_classifies_conventional_asset_directories(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "mmproj").mkdir()
    (tmp_path / "mtp").mkdir()
    (tmp_path / "chat-templates").mkdir()
    for relative in (
        "models/Qwen.gguf",
        "mmproj/mmproj-Qwen.gguf",
        "mtp/Qwen-draft.gguf",
        "chat-templates/qwen.jinja",
    ):
        (tmp_path / relative).write_bytes(b"asset")

    assets, errors = library.scan([tmp_path])

    assert not errors
    assert {asset.relative_path: asset.kind for asset in assets} == {
        "chat-templates/qwen.jinja": "template",
        "mmproj/mmproj-Qwen.gguf": "mmproj",
        "models/Qwen.gguf": "model",
        "mtp/Qwen-draft.gguf": "draft",
    }


def test_scan_reports_bad_roots_and_never_opens_unrelated_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not a model")
    assets, errors = library.scan([tmp_path / "missing"])
    assert assets == []
    assert errors and "missing" in errors[0]


def test_companion_ranking_suggests_without_selecting(tmp_path: Path) -> None:
    model = library.Asset(str(tmp_path / "Qwen3-VL-8B.gguf"), "Qwen3-VL-8B.gguf", str(tmp_path), "model", 1, 0)
    matching = library.Asset(str(tmp_path / "mmproj-Qwen3-VL.gguf"), "mmproj-Qwen3-VL.gguf", str(tmp_path), "mmproj", 1, 0)
    other = library.Asset(str(tmp_path / "mmproj-Llama.gguf"), "mmproj-Llama.gguf", str(tmp_path), "mmproj", 1, 0)
    ranked = library.rank_companions(model, [model, other, matching], "mmproj")
    assert ranked == [matching, other]


def test_host_bundle_fields_render_and_are_validated() -> None:
    profile = StoredProfile(
        name="local", backend="llamacpp", runtime="host", model_path="model.gguf",
        mmproj_path="projector.gguf", draft_model_path="draft.gguf",
        draft_type="draft-mtp", chat_template_path="chat.jinja",
    )
    _validate_profile(profile)
    command = render_command(
        {}, local_model="model.gguf", host="127.0.0.1", port=8080,
        mmproj="projector.gguf", draft_model="draft.gguf", draft_type="draft-mtp",
        chat_template="chat.jinja",
    )
    assert command[command.index("--mmproj") + 1] == "projector.gguf"
    assert command[command.index("--spec-draft-model") + 1] == "draft.gguf"
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert command[command.index("--chat-template-file") + 1] == "chat.jinja"

    embedded = StoredProfile(
        name="embedded", backend="llamacpp", runtime="host", model_path="model.gguf",
        draft_type="draft-mtp", mtp_n=5,
    )
    _validate_profile(embedded)
    embedded_command = render_command(
        {}, local_model="model.gguf", host="127.0.0.1", port=8080,
        draft_type="draft-mtp", mtp_n=5,
    )
    assert "--spec-draft-model" not in embedded_command
    assert embedded_command[embedded_command.index("--spec-type") + 1] == "draft-mtp"
    assert embedded_command[embedded_command.index("--spec-draft-n-max") + 1] == "5"

    with pytest.raises(ValueError, match="unless it is draft-mtp"):
        _validate_profile(StoredProfile(name="bad", backend="llamacpp", runtime="host", draft_type="draft-simple"))
    with pytest.raises(ValueError, match="requires draft_type=draft-mtp"):
        _validate_profile(StoredProfile(name="bad", backend="llamacpp", runtime="host", model_path="model.gguf", mtp_n=2))
    with pytest.raises(ValueError, match="non-negative"):
        _validate_profile(StoredProfile(name="bad", backend="llamacpp", runtime="host", model_path="model.gguf", draft_type="draft-mtp", mtp_n=-1))
