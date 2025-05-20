# client/tui/ws_client.py

import asyncio
import aiohttp
from typing import TYPE_CHECKING, Optional

from shared_models.tui_events import TUIWebSocketMessage
from .messages import ServerUpdate, ConnectionStatusUpdate

if TYPE_CHECKING:
    from textual.app import App # For type hinting the app instance

# Consider moving SERVER_URL to a config file or environment variable later
SERVER_URL = "ws://localhost:5517/ws"
RECONNECT_DELAY_SECONDS = 5


async def listen_to_server(app: "App"): # The Textual App instance will be passed here
    """
    Connects to the Hephia server WebSocket, listens for messages,
    and posts them as ServerUpdate events to the Textual application.
    """
    session: Optional[aiohttp.ClientSession] = None

    while True:
        try:
            if session is None or session.closed:
                # Create a new session if one doesn't exist or the previous one was closed
                session = aiohttp.ClientSession()
                app.post_message(ConnectionStatusUpdate("connecting"))
                app.log(f"WebSocket client: Attempting to connect to {SERVER_URL}...")

            async with session.ws_connect(SERVER_URL) as ws_connection:
                app.post_message(ConnectionStatusUpdate("connected"))
                app.log(f"WebSocket client: Connection established with {SERVER_URL}.")

                async for msg in ws_connection:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            raw_data = msg.data
                            # Deserialize the entire message from the server
                            tui_ws_message = TUIWebSocketMessage.model_validate_json(raw_data)

                            # Post the inner payload to the Textual app as a ServerUpdate event
                            if tui_ws_message.payload:
                                app.post_message(ServerUpdate(tui_ws_message.payload))
                            else:
                                app.log("WebSocket client: Received message with no payload.")

                        except Exception as e:
                            app.log(f"WebSocket client: Error processing message: {e}")
                            app.log(f"WebSocket client: Raw data: {msg.data[:500]}...") # Log problematic data

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        app.log(f"WebSocket client: Connection error reported: {ws_connection.exception()}.")
                        app.post_message(ConnectionStatusUpdate("disconnected", f"Error: {ws_connection.exception()}"))
                        break # Break from inner message loop to trigger reconnection logic

                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        app.log("WebSocket client: Connection closed by server.")
                        app.post_message(ConnectionStatusUpdate("disconnected", "Closed by server"))
                        break # Break from inner message loop to trigger reconnection logic

            # If we exit the 'async with session.ws_connect' block cleanly (e.g., server closed connection gracefully)
            if ws_connection.closed:
                app.post_message(ConnectionStatusUpdate("disconnected", "Connection ended"))
            app.log(f"WebSocket client: Disconnected from {SERVER_URL}. Attempting to reconnect in {RECONNECT_DELAY_SECONDS}s...")

        except aiohttp.ClientError as e: # Handles errors during session.ws_connect
            app.log(f"WebSocket client: Connection attempt failed: {e}. Retrying in {RECONNECT_DELAY_SECONDS}s...")
            app.post_message(ConnectionStatusUpdate("disconnected", str(e)))
        except Exception as e: # Catch-all for other unexpected errors in the loop
            app.log(f"WebSocket client: Unexpected error: {e}. Retrying in {RECONNECT_DELAY_SECONDS}s...")
            app.post_message(ConnectionStatusUpdate("disconnected", f"Unexpected: {str(e)}"))
        finally:
            # Ensure session is closed if it exists and an error caused us to exit the loop before graceful close
            if session and not session.closed:
                await session.close()
            session = None # Force re-creation of session in the next iteration

            await asyncio.sleep(RECONNECT_DELAY_SECONDS)