"""
core/websockets/tui_stream.py

TUI state streaming WebSocket — broadcasts internal state to connected TUI clients.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from shared_models.tui_events import (
    TUIMessage,
    TUISystemContext,
    TUIDataPayload,
    TUIWebSocketMessage,
    TUIMood,
    TUINeed,
    TUIBehavior,
    TUIEmotionalStateItem,
)
from config import Config
from loggers import SystemLogger

if TYPE_CHECKING:
    from core.state_bridge import StateBridge


class TUIStreamManager:
    """Manages WebSocket connections for TUI state broadcasting."""

    def __init__(self, state_bridge: StateBridge) -> None:
        self.state_bridge = state_bridge
        self.active_connections: List[WebSocket] = []
        self.tui_data_lock = asyncio.Lock()
        self.logger = SystemLogger

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Accept a TUI WebSocket connection and stream state updates."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connection accepted. Total active: {len(self.active_connections)}")

        try:
            initial_payload = await self._prepare_tui_data_payload()
            initial_message = TUIWebSocketMessage(
                event_type="TUI_INITIAL_STATE",
                payload=initial_payload,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            await websocket.send_json(initial_message.model_dump())

            while True:
                data = await websocket.receive_json()
                await websocket.send_json({
                    "type": "acknowledgment",
                    "content": "Message received",
                })
        except WebSocketDisconnect:
            self.logger.info("WebSocket client disconnected.")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast_thought_bubble(
        self,
        source: str,
        content: str,
        action: Optional[str] = None,
    ) -> None:
        """Broadcast a thought bubble event to all connected clients."""
        if not self.active_connections:
            return

        message = {
            "event_type": "THOUGHT_BUBBLE",
            "payload": {
                "source": source,
                "content": content,
                "action": action,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.warning(f"Error broadcasting thought bubble: {e}")
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

    async def broadcast_state_update(self) -> None:
        """Broadcast a state refresh to all connected TUI clients."""
        if not self.active_connections:
            return

        try:
            tui_payload = await self._prepare_tui_data_payload()
            tui_message = TUIWebSocketMessage(
                event_type="TUI_REFRESH_DATA",
                payload=tui_payload,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            message_data = tui_message.model_dump()

            for connection in list(self.active_connections):
                try:
                    await connection.send_json(message_data)
                except Exception as e:
                    self.logger.warning(f"Error broadcasting to client: {e}. Removing connection.")
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)
        except Exception as e:
            self.logger.error(f"Failed to broadcast state update: {e}", exc_info=True)

    async def _prepare_tui_data_payload(self) -> TUIDataPayload:
        """Build a TUI data payload from current state bridge data."""
        raw_context = await self.state_bridge.get_api_context(use_memory_emotions=False)

        tui_mood: Optional[TUIMood] = None
        if mood_d := raw_context.get("mood"):
            tui_mood = TUIMood(**mood_d)

        tui_needs: Dict[str, TUINeed] = {}
        if needs_raw := raw_context.get("needs"):
            if isinstance(needs_raw, dict):
                for name, details in needs_raw.items():
                    if isinstance(details, dict) and "satisfaction" in details:
                        tui_needs[name] = TUINeed(satisfaction=float(details["satisfaction"]))

        tui_behavior: Optional[TUIBehavior] = None
        if behavior_d := raw_context.get("behavior"):
            tui_behavior = TUIBehavior(**behavior_d)

        tui_emotional_state: List[TUIEmotionalStateItem] = []
        if emo_state := raw_context.get("emotional_state"):
            if isinstance(emo_state, list):
                for item in emo_state:
                    if isinstance(item, dict):
                        tui_emotional_state.append(TUIEmotionalStateItem(**item))

        system_context = TUISystemContext(
            mood=tui_mood,
            needs=tui_needs if tui_needs else None,
            behavior=tui_behavior,
            emotional_state=tui_emotional_state if tui_emotional_state else None,
        )

        raw_conversation = self.state_bridge.get_latest_raw_conversation_state()
        if not isinstance(raw_conversation, list):
            actual_recent = []
        else:
            actual_recent = raw_conversation[-6:]

        recent_messages = [TUIMessage(**msg) for msg in actual_recent]
        cognitive_summary = self.state_bridge.get_latest_cognitive_summary()

        model_name = "N/A"
        if hasattr(Config, "get_cognitive_model"):
            try:
                model_name = Config.get_cognitive_model() or "N/A"
            except Exception:
                model_name = "N/A"

        return TUIDataPayload(
            recent_messages=recent_messages,
            system_context=system_context,
            cognitive_summary=cognitive_summary,
            current_model_name=model_name,
        )
