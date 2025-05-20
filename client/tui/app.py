# client/tui/app.py

import asyncio
import json
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Header, Footer, Static, RichLog
from textual.binding import Binding
from textual.reactive import reactive
from rich.text import Text
from .ws_client import listen_to_server
from .messages import ServerUpdate, ConnectionStatusUpdate
from shared_models.tui_events import TUIMessage, TUISystemContext, TUIDataPayload


class HephiaTUIApp(App):
    TITLE = "Hephia TUI Client"
    SUB_TITLE = "Real-time System Monitor"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="c", action="clear_logs", description="Clear Logs"),
        Binding(key="alt+1", action="copy_cognitive", description="Copy Cognitive"),
        Binding(key="alt+2", action="copy_state", description="Copy State"),
        Binding(key="alt+3", action="copy_summary", description="Copy Summary"),
    ]


    current_model_name: reactive[str] = "N/A" 
    _original_sub_title: str = "Real-time System Monitor"

    def compose(self) -> ComposeResult:
        """Create child widgets for the app, defining the three-panel layout."""
        yield Header()
        with Horizontal(id="main_layout"):
            with Container(id="cognitive_panel_container"):
                yield Static("Cognitive Processing", classes="panel_title")
                yield RichLog(id="cognitive_content", wrap=True, highlight=True, markup=True, classes="panel_content")
            
            with Vertical(id="right_column"):
                with Container(id="state_panel_container"):
                    yield Static("System State", classes="panel_title")
                    yield Static(id="state_content", classes="panel_content")
                
                with Container(id="summary_panel_container"):
                    yield Static("Cognitive Summary", classes="panel_title")
                    yield RichLog(id="summary_content", wrap=True, markup=True, classes="panel_content")
        yield Static("Status: Initializing...", id="connection_status_bar")
        yield Footer()

    async def on_mount(self) -> None:
        """Called when the app is mounted."""
        self._original_sub_title = self.sub_title
        asyncio.create_task(listen_to_server(self))
        self.log("TUI Client mounted. Attempting to connect to server...")

    def action_clear_logs(self) -> None:
        """Action to clear all log panels."""
        self.query_one("#cognitive_content", RichLog).clear()
        self.query_one("#summary_content", RichLog).clear()
        self.log("Manually cleared TUI panels.")

    async def on_server_update(self, message: ServerUpdate) -> None:
        """Handler for ServerUpdate messages from the WebSocket client."""
        if not message.payload:
            self.log("Received empty payload from server.")
            return

        payload: TUIDataPayload = message.payload

        if payload.current_model_name:
            self.current_model_name = payload.current_model_name

        # 1. Update Cognitive Processing Panel (Recent Messages)
        cog_log = self.query_one("#cognitive_content", RichLog)
        cog_log.clear() 
        if payload.recent_messages:
            for msg in payload.recent_messages:
                role_name = ""
                role_style = ""

                if msg.role == "assistant":
                    role_name = self.current_model_name
                    role_style = "bold green"
                elif msg.role == "user":
                    role_name = "Hephia"
                    role_style = "bold magenta"
                elif msg.role == "system":
                    role_name = "System"
                    role_style = "bold blue"
                else:
                    role_name = msg.role.capitalize()
                    role_style = "dim" # Default for unknown roles

                cog_log.write(f"[{role_style}]{role_name}:[/] {msg.content}")
        else:
            cog_log.write("[i]No recent messages.[/i]")

        # 2. Update System State Panel
        state_widget = self.query_one("#state_content", Static)
        state_lines = []
        if payload.system_context:
            ctx: TUISystemContext = payload.system_context
            
            if ctx.mood:
                state_lines.append(f"[b]Mood:[/b] {ctx.mood.name or 'N/A'} "
                                   f"(V: {ctx.mood.valence:.2f}, A: {ctx.mood.arousal:.2f})")
            else:
                state_lines.append("[b]Mood:[/b] N/A")

            state_lines.append("\n[b]Needs:[/b]")
            if ctx.needs:
                for need_name, need_data in ctx.needs.items():
                    state_lines.append(f"  - {need_name.capitalize()}: {need_data.satisfaction*100:.0f}%")
            else:
                state_lines.append("  N/A")

            state_lines.append("\n[b]Behavior:[/b]")
            if ctx.behavior:
                state_lines.append(f"  {ctx.behavior.name or 'N/A'} "
                                   f"({'Active' if ctx.behavior.active else 'Inactive'})")
            else:
                state_lines.append("  N/A")

            state_lines.append("\n[b]Emotional State:[/b]")
            if ctx.emotional_state:
                for emotion in ctx.emotional_state:
                    state_lines.append(f"  - {emotion.name or 'N/A'}: {emotion.intensity:.2f} "
                                       f"(V: {emotion.valence:.2f}, A: {emotion.arousal:.2f})")
            else:
                state_lines.append("  N/A")
            
            state_widget.update("\n".join(state_lines))
        else:
            state_widget.update("System context not available.")
            
        # 3. Update Cognitive Summary Panel
        summary_log = self.query_one("#summary_content", RichLog)
        summary_log.clear() # Clear previous summary
        if payload.cognitive_summary:
            summary_log.write(payload.cognitive_summary)
        else:
            summary_log.write("[i]No cognitive summary available.[/i]")

    async def on_connection_status_update(self, message: ConnectionStatusUpdate) -> None:
        """Handles connection status updates to display in the status bar."""
        status_bar = self.query_one("#connection_status_bar", Static)

        # Create a Rich Text object to build the styled message
        styled_text_content = Text()

        if message.status == "connected":
            styled_text_content.append("Status: Connected", style="bold #66FF66") # $success
            self.sub_title = self._original_sub_title 
        elif message.status == "connecting":
            styled_text_content.append("Status: Connecting...", style="bold #FFD700") # $warning
            self.sub_title = "Attempting Connection..."
        elif message.status == "disconnected":
            styled_text_content.append("Status: Disconnected", style="bold #D32F2F") # $error
            if message.detail:
                styled_text_content.append(f" - {message.detail}", style="bold #D32F2F")

            # For sub_title, we can use the raw message.detail as it's not parsed by Rich markup
            self.sub_title = f"Disconnected{(' - ' + message.detail) if message.detail else ''}"

        # Update the Static widget with the constructed Text object
        status_bar.update(styled_text_content) 

        self.log(f"UI Connection Status: {message.status}{(' - ' + message.detail) if message.detail else ''}")

    def _copy_to_clipboard(self, content: str, panel_name: str):
        if not hasattr(self.__class__, 'PYPERCLIP_AVAILABLE') or not self.PYPERCLIP_AVAILABLE:
            self.sub_title = "Copy failed: pyperclip library not installed."
            self.bell()
            self.set_timer(3.0, self.action_revert_subtitle) # Use float for delay
            self.log("pyperclip not available for copying.")
            return

        try:
            pyperclip.copy(content)
            self.sub_title = f"Copied '{panel_name}' to clipboard!"
            self.bell() 
        except Exception as e: # Catch generic pyperclip errors
            self.sub_title = f"Copy failed for '{panel_name}'."
            self.log(f"Clipboard error for {panel_name}: {e}")
            self.bell()
        finally:
            self.set_timer(3.0, self.action_revert_subtitle)

    def action_revert_subtitle(self) -> None:
        """Resets the app's sub_title after temporary messages like copy status."""
        self.sub_title = self._original_sub_title

    def _get_plain_text_from_rich_log(self, rich_log_widget: RichLog) -> str:
        """Helper to extract plain text from RichLog's document."""
        plain_text_lines = []
        if rich_log_widget.document: # Check if document exists
            for line_segments in rich_log_widget.document.lines: # List[List[Segment]]
                plain_text_lines.append("".join(segment.text for segment in line_segments))
        return "\n".join(plain_text_lines)

    def action_copy_cognitive(self) -> None:
        content_log = self.query_one("#cognitive_content", RichLog)
        text_content = self._get_plain_text_from_rich_log(content_log)
        self._copy_to_clipboard(text_content, "Cognitive Log")

    def action_copy_state(self) -> None:
        content_widget = self.query_one("#state_content", Static)
        # Static widget's renderable is often a Rich object (like Text)
        if hasattr(content_widget.renderable, 'plain'):
            text_content = content_widget.renderable.plain
        else: # Fallback if it's just a string
            text_content = str(content_widget.renderable)
        self._copy_to_clipboard(text_content, "System State")

    def action_copy_summary(self) -> None:
        content_log = self.query_one("#summary_content", RichLog)
        text_content = self._get_plain_text_from_rich_log(content_log)
        self._copy_to_clipboard(text_content, "Cognitive Summary")

try:
    import pyperclip
    HephiaTUIApp.PYPERCLIP_AVAILABLE = True # Set as class attribute
except ImportError:
    HephiaTUIApp.PYPERCLIP_AVAILABLE = False
    print("WARNING: pyperclip library not found. Copy to clipboard functionality will be disabled.")

