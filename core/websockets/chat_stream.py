"""
core/websockets/chat_stream.py

Bidirectional WebSocket for chat — alternative to POST /chat for persistent connections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from loggers import SystemLogger

if TYPE_CHECKING:
    from mind.mind import Mind


class ChatStreamManager:
    """Handles /ws/chat WebSocket connections."""

    def __init__(self, mind: Mind) -> None:
        self.mind = mind

    async def handle_connection(self, websocket: WebSocket) -> None:
        """
        Accept a chat WebSocket connection.

        Protocol:
            Client → {"message": "Hello!"}
            Server → {"role": "assistant", "content": "...", "was_task": false, "memories_used": [...]}
        """
        await websocket.accept()
        SystemLogger.info("Chat WebSocket connection accepted.")

        try:
            while True:
                data = await websocket.receive_json()
                message = data.get("message")
                if not message:
                    await websocket.send_json({
                        "error": "Missing 'message' field",
                    })
                    continue

                try:
                    response = await self.mind.process_message(message)
                    await websocket.send_json({
                        "role": "assistant",
                        "content": response.content,
                        "was_task": response.was_task,
                        "memories_used": response.memories_used,
                    })
                except Exception as e:
                    SystemLogger.error(f"Chat WebSocket processing error: {e}")
                    await websocket.send_json({
                        "error": f"Processing failed: {e}",
                    })
        except WebSocketDisconnect:
            SystemLogger.info("Chat WebSocket client disconnected.")
        except Exception as e:
            SystemLogger.error(f"Chat WebSocket error: {e}")
