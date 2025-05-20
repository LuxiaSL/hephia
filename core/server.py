"""
Core server implementation for Hephia.

This server acts as the central nervous system, coordinating communication
between the internal systems, LLM brain, and external interfaces.
It manages both HTTP endpoints for commands and WebSocket connections
for real-time state updates.
"""
from __future__ import annotations
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any, Optional
import asyncio
from pydantic import BaseModel, Field

from brain.cognition.notification import Notification
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

from shared_models.tui_events import (
    TUIMessage,
    TUISystemContext,
    TUIDataPayload, 
    TUIWebSocketMessage,
    TUIMood,
    TUINeed,
    TUIBehavior,
    TUIEmotionalStateItem
)

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
    channel_id: str  # Keep for backward compatibility
    message_id: str
    author: str
    author_id: str
    content: str
    timestamp: str
    context: Dict[str, Any]

class DiscordChannelUpdate(BaseModel):
    channel_id: str  # Keep for backward compatibility
    channel_name: str
    guild_name: Optional[str] = None
    new_message_count: int
    
    @property
    def path(self) -> str:
        """Generate channel path from guild and channel names."""
        return f"{self.guild_name}:{self.channel_name}" if self.guild_name else self.channel_name

class ActionRequest(BaseModel):
    """Model for action execution requests."""
    action: str = Field(..., description="Name of the action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Optional parameters for the action")
    message: Optional[str] = Field(None, description="Optional message to attach to memory/notification")

class ActionResponse(BaseModel):
    """Model for action execution responses."""
    success: bool
    message: str
    state_changes: Dict[str, Any]
    memory_id: Optional[str] = None

class ActionStatus(BaseModel):
    """Model for action status information."""
    last_execution: float
    total_executions: int
    successful_executions: int
    failed_executions: int
    on_cooldown: bool
    remaining_cooldown: float

class ActionInfo(BaseModel):
    """Model for action information."""
    name: str
    description: str
    parameters: Dict[str, Any]
    status: ActionStatus

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

        self.latest_recent_messages_for_tui: List[TUIMessage] = []
        self.latest_system_context_for_tui: Optional[TUISystemContext] = None
        self.latest_cognitive_summary_for_tui: str = ""
        self.tui_data_lock = asyncio.Lock()

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
            action_manager=instance.internal.action_manager,
            discord_service=instance.discord_service
        )
        instance.core_processor = CoreProcessor(
            api_manager=instance.api,
            state_bridge=instance.state_bridge,
            cognitive_bridge=instance.internal.cognitive_bridge,
            environment_registry=instance.environment_registry,
            discord_service=instance.discord_service,
        )
        # Now that all components are in place, setup routes and event handlers.
        instance.setup_routes()
        instance.setup_event_handlers()
        return instance
    
    def setup_middleware(self) -> None:
        """Configure server middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
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
            await self.core_processor.prune_conversation()
            return {"status": "success"}
        
        @self.app.post("/discord_inbound")
        async def discord_inbound(payload: DiscordInbound):
            try:
                SystemLogger.info("Received new Discord message.")
            
                # Extract channel and guild information from context
                channel_name = payload.context.get('channel_name', 'Unknown')
                guild_name = payload.context.get('guild_name')
                
                # Create path-based channel reference
                channel_path = f"{guild_name}:{channel_name}" if guild_name else channel_name
                
                # Prepare context for the AI with path-based references
                message_context = {
                    'current_message': {
                        'id': payload.message_id,
                        'author': payload.author,  # Use clean author name instead of mention format
                        'author_id': payload.author_id,  # Keep ID for reference if needed
                        'content': payload.content,
                        'timestamp': payload.timestamp
                    },
                    'channel': {
                        'id': payload.channel_id,  # Keep ID for backward compatibility
                        'name': channel_name,
                        'guild_name': guild_name,
                        'path': channel_path,  # Add path for clean references
                        'recent_activity': payload.context.get('message_count')
                    },
                    'conversation_history': payload.context.get('recent_history', [])
                }
                SystemLogger.debug(f"Discord message context: {message_context}")
                response_text = await self.core_processor.handle_discord_message(message_context)
                SystemLogger.debug(f"AI response: {response_text[:100]}...")

                if response_text == "":
                    SystemLogger.warning("Received empty response from AI.")
                    return {"status": "no_response"}

                # Use path-based message sending instead of ID-based
                await self.discord_service.queue_message_by_path(
                    path=channel_path,
                    content=response_text
                )

                return {"status": "ok"}
            
            except Exception as e:
                SystemLogger.error(f"Error processing Discord message: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/discord_channel_update")
        async def discord_channel_update(payload: DiscordChannelUpdate):
            # Use guild_name from the payload for the path-based approach
            await self.core_processor.handle_discord_channel_update(
                channel_id=payload.channel_id,
                channel_name=payload.channel_name,
                guild_name=payload.guild_name,
                message_count=payload.new_message_count
            )
            return {"status": "ok"}
        
        @self.app.get("/v1/actions/state")
        async def get_state():
            """Get current system state."""
            return await self.state_bridge.get_api_context(use_memory_emotions=False)

        @self.app.post("/v1/actions/{action_name}")
        async def execute_action(
            action_name: str,
            request: ActionRequest
        ) -> ActionResponse:
            """Execute a specific action."""
            try:
                # Validate action exists
                if action_name not in self.internal.action_manager.available_actions:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Action '{action_name}' not found"
                    )

                # Execute action
                result = self.internal.action_manager.perform_action(
                    action_name,
                    **request.parameters
                )

                if not result["success"]:
                    raise HTTPException(
                        status_code=400,
                        detail=result.get("error", "Action failed")
                    )

                # Create notification for cognitive awareness
                notification = await self.core_processor.handle_action_request({
                    "type": "action_execution",
                    "action": action_name,
                    "result": result,
                    "message": request.message,
                    "timestamp": datetime.now(),
                    "state_changes": result.get("state_changes", {})
                })

                 # Request memory formation if we have a message or significant state change
                if result.get("state_changes"):
                    # Request memory formation through cognitive bridge
                    global_event_dispatcher.dispatch_event(Event(
                        "cognitive:memory:request_formation",
                        {
                            "event_type": "action_execution",
                            "event_data": {
                                "content": self.core_processor.interfaces['action']._generate_summary([notification]),
                                "context": {
                                    "action": action_name,
                                    "result": result,
                                    "state_changes": result.get("state_changes", {}),
                                    "user_message": request.message
                                }
                            }
                        }
                    ))

                # Dispatch event for state update
                global_event_dispatcher.dispatch_event(Event("internal:action", {
                    "action": action_name,
                    "result": result,
                    "has_memory": bool(request.message or result.get("state_changes"))
                }))

                return ActionResponse(
                    success=True,
                    message=f"Successfully executed {action_name}",
                    state_changes=result.get("state_changes", {}),
                    memory_created=bool(result.get("state_changes"))
                )

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/v1/actions")
        async def get_available_actions() -> Dict[str, ActionInfo]:
            """Get information about all available actions and their current status."""
            try:
                # Get actions and their current status
                actions = self.internal.action_manager.available_actions
                statuses = self.internal.action_manager.get_action_status()
                
                response = {}
                for name, action in actions.items():
                    # Extract parameter info from action class
                    parameters = {}
                    if hasattr(action, '_base_cooldown'):
                        parameters['cooldown'] = action._base_cooldown
                    if hasattr(action, 'calculate_recovery_amount'):
                        parameters['dynamic_recovery'] = True
                    
                    # Get current status and flatten it.
                    raw_status = statuses.get(name, {})
                    flat_status = self.flatten_action_status(raw_status)
                    status_obj = ActionStatus(**flat_status)
                    
                    response[name] = ActionInfo(
                        name=name,
                        description=action.__doc__ or "No description available",
                        parameters=parameters,
                        status=status_obj
                    )
                
                return response
                        
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


        @self.app.get("/v1/actions/{action_name}/status")
        async def get_action_status(action_name: str) -> ActionStatus:
            """Get detailed status information for a specific action."""
            try:
                if action_name not in self.internal.action_manager.available_actions:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Action '{action_name}' not found"
                    )

                raw_status = self.internal.action_manager.get_action_status(action_name)
                flat_status = self.flatten_action_status(raw_status)
                return ActionStatus(**flat_status)

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def flatten_action_status(self, raw_status: dict) -> dict:
        """
        Helper to flatten the status dictionary returned by the ActionManager.
        Expected raw_status structure:
        {
            "history": {
                "last_execution": ...,
                "total_executions": ...,
                "successful_executions": ...,
                "failed_executions": ...
            },
            "on_cooldown": <bool>,
            "remaining_cooldown": <float>
        }
        This returns a flat dict that matches the ActionStatus model.
        """
        history = raw_status.get('history', {})
        return {
            'last_execution': history.get('last_execution', 0),
            'total_executions': history.get('total_executions', 0),
            'successful_executions': history.get('successful_executions', 0),
            'failed_executions': history.get('failed_executions', 0),
            'on_cooldown': raw_status.get('on_cooldown', False),
            'remaining_cooldown': raw_status.get('remaining_cooldown', 0)
        }

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
        """Handle individual WebSocket connections."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connection accepted. Total active: {len(self.active_connections)}")
        
        try:
            self.logger.info("Preparing initial TUI data for new WebSocket client.")
            initial_payload = await self._prepare_tui_data_payload()
            
            initial_tui_message = TUIWebSocketMessage(
                event_type="TUI_INITIAL_STATE",
                payload=initial_payload,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

            await websocket.send_json(initial_tui_message.model_dump())
            self.logger.info("Initial TUI data sent to new WebSocket client.")

            while True:
                data = await websocket.receive_json()
                await self.handle_websocket_message(websocket, data) 
                
        except WebSocketDisconnect:
            self.logger.info(f"WebSocket client disconnected. Total active: {len(self.active_connections) -1}")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}", exc_info=True)
        finally: # Ensure removal on any exit from the try block
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket connection closed. Total active: {len(self.active_connections)}")

    
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
        Prepares and broadcasts a comprehensive TUI state update to all connected WebSocket clients.
        This is typically triggered when a significant state change (like 'state:changed') has occurred.
        """
        if not self.active_connections:
            return

        self.logger.debug(f"Preparing TUI broadcast due to server event: {event.event_type}")
        
        try:
            # Prepare the full TUI data payload using the latest data.
            tui_payload = await self._prepare_tui_data_payload()
            
            tui_websocket_message = TUIWebSocketMessage(
                event_type="TUI_REFRESH_DATA",
                payload=tui_payload,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            message_to_send = tui_websocket_message.model_dump()
            
            # Iterate over a copy of the list because removal can happen during iteration
            for connection in list(self.active_connections):
                try:
                    await connection.send_json(message_to_send)
                except Exception as e:
                    self.logger.warning(f"Error broadcasting TUI data to client: {e}. Removing connection.")
                    if connection in self.active_connections: # Check again, could be removed by another task
                        self.active_connections.remove(connection)
            
            self.logger.debug(f"TUI data broadcast to {len(self.active_connections)} WebSocket clients.")

        except Exception as e:
            self.logger.error(f"Failed to prepare or broadcast TUI data: {e}", exc_info=True)

    async def _prepare_tui_data_payload(self) -> TUIDataPayload:
        """
        Fetches the latest data components from StateBridge,
        updates the server's TUI cache, and constructs the TUIDataPayload.
        """

        # 1. Get and process System Context
        raw_context_data = await self.state_bridge.get_api_context(use_memory_emotions=False)
        
        tui_mood: Optional[TUIMood] = None
        if mood_d := raw_context_data.get('mood'):
            tui_mood = TUIMood(**mood_d)

        tui_needs: Dict[str, TUINeed] = {}
        if needs_d_raw := raw_context_data.get('needs'):
            if isinstance(needs_d_raw, dict):
                for need_name, need_details in needs_d_raw.items():
                    if isinstance(need_details, dict) and 'satisfaction' in need_details:
                        tui_needs[need_name] = TUINeed(satisfaction=float(need_details['satisfaction']))

        tui_behavior: Optional[TUIBehavior] = None
        if behavior_d := raw_context_data.get('behavior'):
            tui_behavior = TUIBehavior(**behavior_d)

        tui_emotional_state: List[TUIEmotionalStateItem] = []
        if emo_state_d := raw_context_data.get('emotional_state'):
            if isinstance(emo_state_d, list):
                for item in emo_state_d:
                    if isinstance(item, dict):
                        tui_emotional_state.append(TUIEmotionalStateItem(**item))
        
        current_system_context = TUISystemContext(
            mood=tui_mood,
            needs=tui_needs if tui_needs else None, # Ensure None if empty
            behavior=tui_behavior,
            emotional_state=tui_emotional_state if tui_emotional_state else None # Ensure None if empty
        )

        # 2. Get and process Recent Messages (from raw_state)
        raw_conversation_data = self.state_bridge.get_latest_raw_conversation_state()
        
        num_recent_messages_to_send = 6
        
        # Ensure raw_conversation_data is a list before slicing
        if not isinstance(raw_conversation_data, list):
            actual_recent_raw = []
            self.logger.warning("raw_conversation_data from StateBridge was not a list.")
        else:
            actual_recent_raw = raw_conversation_data[-num_recent_messages_to_send:]
        
        current_recent_messages = [TUIMessage(**msg) for msg in actual_recent_raw]

        # 3. Get Cognitive Summary
        current_cognitive_summary = self.state_bridge.get_latest_cognitive_summary()

        model_name_from_config = "N/A" # Default
        if hasattr(Config, 'get_cognitive_model'): # Check if method exists
            try:
                model_name_from_config = Config.get_cognitive_model()
                if not model_name_from_config: # Handle empty return
                    model_name_from_config = "N/A"
            except Exception as e:
                self.logger.warning(f"Could not retrieve model name from Config: {e}")
                model_name_from_config = "N/A" # Fallback

        async with self.tui_data_lock:
            self.latest_system_context_for_tui = current_system_context
            self.latest_recent_messages_for_tui = current_recent_messages
            self.latest_cognitive_summary_for_tui = current_cognitive_summary

        return TUIDataPayload(
            recent_messages=current_recent_messages,
            system_context=current_system_context,
            cognitive_summary=current_cognitive_summary,
            current_model_name=model_name_from_config
        )

    
    def run(self, host: str = "0.0.0.0", port: int = 5517):
        """
        Run the server.
        
        Args:
            host: Host to run on
            port: Port to run on
        """
        uvicorn.run(self.app, host=host, port=port)