"""Read-only local asset picker for host llama.cpp profiles."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList, Static
from textual.widgets.option_list import Option

from tui.backends.llamacpp import library
from tui.common.i18n import t


class AssetPickerScreen(ModalScreen[str | None]):
    """Choose one explicit asset from configured library roots."""

    BINDINGS = [Binding("escape", "cancel", show=False)]
    DEFAULT_CSS = """
    AssetPickerScreen { align: center middle; }
    AssetPickerScreen > Vertical {
        background: $surface; border: round $primary; padding: 1 2;
        width: 90%; max-width: 96; height: 80%;
    }
    AssetPickerScreen #asset-title { text-style: bold; color: $primary; }
    AssetPickerScreen #asset-hint { color: $text-muted; margin-bottom: 1; }
    AssetPickerScreen OptionList { height: 1fr; }
    """

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def compose(self) -> ComposeResult:
        assets, errors = library.scan()
        selected = [asset for asset in assets if asset.kind == self.kind]
        options = [
            Option(f"{asset.relative_path}  [dim]{library.format_size(asset.size_bytes)}[/dim]", id=asset.path)
            for asset in selected
        ]
        if not options:
            options = [Option(t("No matching assets found", "일치하는 자산이 없습니다"), disabled=True)]
        with Vertical():
            yield Static(t(f"Choose {self.kind} asset", f"{self.kind} 자산 선택"), id="asset-title")
            hint = t("Library roots are read from defaults.llamacpp.library_roots.",
                     "라이브러리 루트는 defaults.llamacpp.library_roots에서 읽습니다.")
            if errors:
                hint += " " + "; ".join(errors)
            yield Label(hint, id="asset-hint")
            yield OptionList(*options, id="asset-list")

    @on(OptionList.OptionSelected, "#asset-list")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)
