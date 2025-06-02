# client/config/__main__.py
"""
Main entry point for the Hephia Configuration TUI.
Allows running the TUI as a package: python -m client.config
"""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane, Label
from textual.binding import Binding

from .screens.env_editor import EnvEditorScreen
from .screens.models_editor import ModelsEditorScreen
from .screens.prompt_editor import PromptEditorScreen


class ConfigApp(App):
    """Hephia Configuration Management TUI"""

    TITLE = "Hephia Configuration Tool"
    CSS_PATH = "app.tcss" # We'll need a basic TCSS file

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    hide_keys: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-env"):
            with TabPane("Environment (.env)", id="tab-env"):
                yield EnvEditorScreen()
            with TabPane("Custom Models (models.json)", id="tab-models"):
                yield ModelsEditorScreen()
            with TabPane("YAML Prompts", id="tab-prompts"):
                yield PromptEditorScreen()
        yield Footer()

def run() -> None:
    app = ConfigApp()
    app.run()

if __name__ == "__main__":
    run()