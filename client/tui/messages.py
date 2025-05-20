# client/tui/messages.py

from textual.message import Message as TextualMessage # Alias to avoid name collision
from typing import Optional

from shared_models.tui_events import TUIDataPayload


class ServerUpdate(TextualMessage):
    """
    Custom Textual message to signal that new data has been received
    from the Hephia server for the TUI.
    """
    def __init__(self, payload: Optional[TUIDataPayload]) -> None:
        self.payload: Optional[TUIDataPayload] = payload
        super().__init__()

class ConnectionStatusUpdate(TextualMessage):
    """Custom Textual message to signal a change in WebSocket connection status."""
    def __init__(self, status: str, detail: Optional[str] = None) -> None:
        self.status: str = status
        self.detail: Optional[str] = detail
        super().__init__()