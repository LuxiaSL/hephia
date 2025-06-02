# client/config/dialogs.py
"""
Reusable dialogs for the Hephia Configuration TUI.
"""
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static
from textual.containers import Vertical, Horizontal

class ConfirmationDialog(ModalScreen[bool]):
    """A modal dialog to confirm an action from the user."""
    def __init__(
        self,
        prompt: str = "Are you sure?",
        confirm_button_label: str = "Yes",
        confirm_button_variant: str = "primary",
        cancel_button_label: str = "No",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name, id, classes)
        self.prompt_text = prompt
        self.confirm_button_label = confirm_button_label
        self.confirm_button_variant = confirm_button_variant
        self.cancel_button_label = cancel_button_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirmation_dialog_content", classes="modal-dialog"):
            yield Static(self.prompt_text, id="confirmation_prompt", classes="dialog-prompt")
            with Horizontal(id="confirmation_buttons", classes="dialog-buttons"):
                yield Button(self.confirm_button_label, variant=self.confirm_button_variant, id="confirm")
                yield Button(self.cancel_button_label, variant="default", id="cancel")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss(True)
        elif event.button.id == "cancel":
            self.dismiss(False)