"""
Core server implementation for Hephia.

This server acts as the central nervous system, coordinating communication
between the internal systems, LLM brain, and external interfaces.
It manages both HTTP endpoints for commands and WebSocket connections
for real-time state updates.
"""
from __future__ import annotations
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any
import asyncio
from pydantic import BaseModel

from core.timer import TimerCoordinator
from core.state_bridge import StateBridge
from core.event_bridge import EventBridge
from core.discord_service import DiscordService
from event_dispatcher import global_event_dispatcher, Event
from internal.internal import Internal
from config import Config
from brain.core.processor import CoreProcessor
from brain.environments.environment_registry import EnvironmentRegistry
from api_clients import APIManager
from loggers import SystemLogger

class CommandRequest(BaseModel):
    """Model for incoming commands."""
    messages: List[Dict[str, str]]
    stream: bool = False

class StateUpdateResponse(BaseModel):
    """Model for state updates."""
    event_type: str
    data: Dict[str, Any]

class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False

class DiscordInbound(BaseModel):
    channel_id: str
    message_id: str
    author: str
    author_id: str
    content: str
    timestamp: str
    context: Dict[str, Any]

class DiscordChannelUpdate(BaseModel):
    channel_id: str
    channel_name: str
    new_message_count: int

class HephiaServer:
    """
    Main server class coordinating all Hephia's systems.
    """
    
    def __init__(self) -> None:
        """Synchronous constructor; assign default or placeholder values."""
        self.app = FastAPI(title="Hephia Server")
        self.setup_middleware()
        self.active_connections: List[WebSocket] = []
        self.timer = TimerCoordinator()
        self.discord_service = DiscordService()

        self.api = APIManager.from_env()
        # Placeholders for async components.
        self.internal = None  # type: Internal
        self.state_bridge = None  # type: StateBridge
        self.event_bridge = None  # type: EventBridge
        self.environment_registry = None  # type: EnvironmentRegistry
        self.core_processor = None # type: CoreProcessor

        self.logger = SystemLogger

    @classmethod
    async def create(cls) -> HephiaServer:
        """
        Asynchronously create and initialize a HephiaServer instance.
        """
        instance = cls()
        # Initialize async components:
        instance.internal = await Internal.create(instance.api)
        instance.state_bridge = StateBridge(internal=instance.internal)
        instance.event_bridge = EventBridge(state_bridge=instance.state_bridge)
        instance.environment_registry = EnvironmentRegistry(
            instance.api,
            cognitive_bridge=instance.internal.cognitive_bridge,
            discord_service=instance.discord_service
        )
        instance.core_processor = CoreProcessor(
            api_manager=instance.api,
            state_bridge=instance.state_bridge,
            cognitive_bridge=instance.internal.cognitive_bridge,
            environment_registry=instance.environment_registry,
        )
        # Now that all components are in place, setup routes and event handlers.
        instance.setup_routes()
        instance.setup_event_handlers()
        return instance
    
    def setup_middleware(self) -> None:
        """Configure server middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Adjust for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self) -> None:
        """Configure server routes."""
        
        @self.app.on_event("startup")
        async def startup():
            """Initialize systems on server startup."""
            await self.startup()
            
        @self.app.on_event("shutdown")
        async def shutdown():
            """Clean up on server shutdown."""
            await self.shutdown()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Handle WebSocket connections for real-time updates."""
            await self.handle_websocket_connection(websocket)
        
        @self.app.post("/v1/chat/completions")
        async def handle_conversation(request: ChatRequest = Body(...)):
            """
            A conversation endpoint that accepts a full conversation history from the client
            and returns the next assistant response.
            """
            if not request.messages:
                raise HTTPException(status_code=400, detail="No messages provided")

            # Prepare messages in the format expected by the core processor
            messages = []
            for msg in request.messages:
                if not msg.role or not msg.content:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid message format - missing role or content"
                    )
                messages.append({"role": msg.role, "content": msg.content})

            try:
                # Log and notify about the incoming message snippet
                snippet = messages[-1]['content'][:1000] if messages else "No content"
                SystemLogger.info(f"Talking with your current user: {snippet}")
            except Exception as e:
                SystemLogger.error(f"Error processing message snippet: {e}")

            try:
                # Process the chat completion request using the core processor
                response = await self.core_processor.handle_chat_completion(messages)
                return response
            except Exception as e:
                SystemLogger.error(f"Error processing conversation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            
        @self.app.post("/v1/prune_conversation")
        async def prune_conversation():
            """Prune the conversation history."""
            self.exo_processor.prune_conversation()
            return {"status": "success"}
            
        @self.app.get("/state")
        async def get_state():
            """Get current system state."""
            return await self.state_bridge.get_api_context(use_memory_emotions=False)
        
        @self.app.post("/discord_inbound")
        async def discord_inbound(payload: DiscordInbound):
            try:
                SystemLogger.info("Received new Discord message.")
            
                # Prepare context for the AI
                message_context = {
                    'current_message': {
                        'id': payload.message_id,
                        'author': f"<@{payload.author_id}>",  # Discord mention format
                        'author_id': payload.author_id,
                        'content': payload.content,
                        'timestamp': payload.timestamp
                    },
                    'channel': {
                        'id': payload.channel_id,
                        'name': payload.context.get('channel_name'),
                        'guild_name': payload.context.get('guild_name'),
                        'recent_activity': payload.context.get('message_count')
                    },
                    'conversation_history': payload.context.get('recent_history', [])
                }
                SystemLogger.debug(f"Discord message context: {message_context}")
                response_text = await self.core_processor.handle_discord_message(message_context)
                SystemLogger.debug(f"AI response: {response_text[:100]}...")

                await self.discord_service.send_message(
                    channel_id=payload.channel_id,
                    content=response_text
                )

                return {"status": "ok"}
            
            except Exception as e:
                SystemLogger.error(f"Error processing Discord message: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/discord_channel_update")
        async def discord_channel_update(payload: DiscordChannelUpdate):
            await self.core_processor.handle_discord_channel_update(
                channel_id=payload.channel_id,
                channel_name=payload.channel_name
            )
            return {"status": "ok"}
    
    def setup_event_handlers(self) -> None:
        """Set up handlers for system events."""
        # Create tasks for async handlers
        global_event_dispatcher.add_listener(
            "state:changed",
            lambda event: asyncio.create_task(self.broadcast_state_update(event))
        )
        global_event_dispatcher.add_listener(
            "internal:action",
            lambda event: asyncio.create_task(self.broadcast_state_update(event))
        )
    
    async def startup(self):
        """Initialize all systems in correct order."""
        try:
            # retrieve & apply prior state first
            SystemLogger.info("Initializing state bridge...")
            await self.state_bridge.initialize()
            
            # init exo
            SystemLogger.info("Initializing processor...")
            await self.core_processor.initialize()
            SystemLogger.info("Starting processor...")
            await self.core_processor.start()

            # init discord
            SystemLogger.info("Initializing Discord service...")
            await self.discord_service.initialize()
            
            # simple periodic internal timers
            SystemLogger.info("Adding internal timers...")
            self.timer.add_task(
                name="needs_update",
                interval=Config.NEED_UPDATE_TIMER,
                callback=self.internal.update_needs
            )
            
            self.timer.add_task(
                name="emotions_update",
                interval=Config.EMOTION_UPDATE_TIMER,
                callback=self.internal.update_emotions
            )
            
            # Start timer
            SystemLogger.info("Starting timer...")
            asyncio.create_task(self.timer.run())
        except Exception as e:
            await self.shutdown()
            raise RuntimeError(f"Startup failed: {str(e)}") from e
    
    async def shutdown(self):
        """Clean shutdown of all systems."""
        try:
            self.timer.stop()
            await self.internal.stop()
            await self.core_processor.stop()
            await self.state_bridge._save_session()
            await self.discord_service.cleanup()

        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise

    async def handle_websocket_connection(self, websocket: WebSocket):
        """
        Handle individual WebSocket connections.
        
        Args:
            websocket: The WebSocket connection to handle
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            # Send initial state
            initial_state = await self.state_bridge.get_api_context(use_memory_emotions=False)
            await websocket.send_json({
                "type": "initial_state",
                "data": initial_state
            })
            
            # Handle incoming messages
            while True:
                data = await websocket.receive_json()
                await self.handle_websocket_message(websocket, data)
                
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def handle_websocket_message(self, websocket: WebSocket, message: Dict):
        """
        Handle incoming WebSocket messages.
        
        Future functionality:
        - Command routing for direct user interactions
        - State update requests
        - Conversation history requests
        - Exo loop visualization updates
        
        Current placeholder until terminal UI implementation.
        
        Args:
            websocket: The WebSocket connection
            message: The message to handle
        """
        # TODO: Implement full message handling for terminal UI
        # For now, just acknowledge receipt
        await websocket.send_json({
            "type": "acknowledgment",
            "content": "Message received - full functionality coming soon"
        })
    
    async def broadcast_state_update(self, event: Event):
        """
        Broadcast state updates to all connected clients.
        
        Args:
            event: The event that triggered the update
        """
        if not self.active_connections:
            return
            
        update = StateUpdateResponse(
            event_type=event.event_type,
            data=event.data
        )
        
        for connection in self.active_connections:
            try:
                await connection.send_json(update.dict())
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                self.active_connections.remove(connection)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the server.
        
        Args:
            host: Host to run on
            port: Port to run on
        """
        uvicorn.run(self.app, host=host, port=port)