"""
Core server implementation for Hephia (Pet).

Stripped of brain/ and discord dependencies.
This is a transitional stub — will be fully rewritten for the pet architecture.
See pet_implementation_spec.md Part 2.6 for the target API surface.
"""
from __future__ import annotations
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any, Optional
import asyncio
from pydantic import BaseModel, Field

from core.timer import TimerCoordinator
from core.state_bridge import StateBridge
from core.event_bridge import EventBridge
from event_dispatcher import global_event_dispatcher, Event
from internal.internal import Internal
from config import Config
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


# --- Pydantic Models (kept for API contract) ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False

class StateUpdateResponse(BaseModel):
    event_type: str
    data: Dict[str, Any]

class ActionRequest(BaseModel):
    action: str = Field(..., description="Name of the action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = Field(None)

class ActionResponse(BaseModel):
    success: bool
    message: str
    state_changes: Dict[str, Any]

class ActionStatus(BaseModel):
    last_execution: float
    total_executions: int
    successful_executions: int
    failed_executions: int
    on_cooldown: bool
    remaining_cooldown: float

class ActionInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    status: ActionStatus


# --- Server ---

class HephiaServer:
    """
    Transitional server stub.
    Initializes internal state + memory systems, exposes state and action endpoints.
    Chat/mind layer not yet wired (pending Phase 3).
    """

    def __init__(self) -> None:
        self.app = FastAPI(title="Hephia Pet Server")
        self.setup_middleware()
        self.active_connections: List[WebSocket] = []
        self.timer = TimerCoordinator()
        self.api = APIManager.from_env()

        # Async components (set in create())
        self.internal: Optional[Internal] = None
        self.state_bridge: Optional[StateBridge] = None
        self.event_bridge: Optional[EventBridge] = None

        self.logger = SystemLogger

        self.latest_recent_messages_for_tui: List[TUIMessage] = []
        self.latest_system_context_for_tui: Optional[TUISystemContext] = None
        self.latest_cognitive_summary_for_tui: str = ""
        self.tui_data_lock = asyncio.Lock()

    @classmethod
    async def create(cls) -> HephiaServer:
        """Asynchronously create and initialize a HephiaServer instance."""
        instance = cls()
        instance.internal = await Internal.create(instance.api)
        instance.state_bridge = StateBridge(internal=instance.internal)
        instance.event_bridge = EventBridge(state_bridge=instance.state_bridge)
        instance.setup_routes()
        instance.setup_event_handlers()
        return instance

    def setup_middleware(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self) -> None:
        @self.app.on_event("startup")
        async def startup():
            await self.startup()

        @self.app.on_event("shutdown")
        async def shutdown():
            await self.shutdown()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket_connection(websocket)

        @self.app.post("/v1/chat/completions")
        async def handle_conversation(request: ChatRequest = Body(...)):
            """Chat endpoint — stub until mind layer is built."""
            # TODO: Wire to mind layer in Phase 3
            raise HTTPException(
                status_code=501,
                detail="Chat not yet implemented — pending mind layer (Phase 3)"
            )

        @self.app.get("/v1/actions/state")
        async def get_state():
            """Get current system state."""
            return await self.state_bridge.get_api_context(use_memory_emotions=False)

        @self.app.post("/v1/actions/{action_name}")
        async def execute_action(action_name: str, request: ActionRequest) -> ActionResponse:
            """Execute a specific action."""
            try:
                if action_name not in self.internal.action_manager.available_actions:
                    raise HTTPException(status_code=404, detail=f"Action '{action_name}' not found")

                result = self.internal.action_manager.perform_action(action_name, **request.parameters)

                if not result["success"]:
                    raise HTTPException(status_code=400, detail=result.get("error", "Action failed"))

                global_event_dispatcher.dispatch_event(Event("internal:action", {
                    "action": action_name,
                    "result": result,
                }))

                return ActionResponse(
                    success=True,
                    message=f"Successfully executed {action_name}",
                    state_changes=result.get("state_changes", {}),
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/v1/actions")
        async def get_available_actions() -> Dict[str, ActionInfo]:
            """Get information about all available actions and their current status."""
            try:
                actions = self.internal.action_manager.available_actions
                statuses = self.internal.action_manager.get_action_status()

                response = {}
                for name, action in actions.items():
                    parameters = {}
                    if hasattr(action, '_base_cooldown'):
                        parameters['cooldown'] = action._base_cooldown
                    if hasattr(action, 'calculate_recovery_amount'):
                        parameters['dynamic_recovery'] = True

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
            try:
                if action_name not in self.internal.action_manager.available_actions:
                    raise HTTPException(status_code=404, detail=f"Action '{action_name}' not found")
                raw_status = self.internal.action_manager.get_action_status(action_name)
                flat_status = self.flatten_action_status(raw_status)
                return ActionStatus(**flat_status)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def flatten_action_status(self, raw_status: dict) -> dict:
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
        global_event_dispatcher.add_listener(
            "state:changed",
            lambda event: asyncio.create_task(self.broadcast_state_update(event))
        )
        global_event_dispatcher.add_listener(
            "internal:action",
            lambda event: asyncio.create_task(self.broadcast_state_update(event))
        )

    async def startup(self):
        try:
            SystemLogger.info("Initializing state bridge...")
            await self.state_bridge.initialize()

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

            SystemLogger.info("Starting timer...")
            asyncio.create_task(self.timer.run())
        except Exception as e:
            await self.shutdown()
            raise RuntimeError(f"Startup failed: {str(e)}") from e

    async def shutdown(self):
        try:
            self.timer.stop()
            await self.internal.stop()
            await self.state_bridge._save_session()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            raise

    async def handle_websocket_connection(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connection accepted. Total active: {len(self.active_connections)}")

        try:
            initial_payload = await self._prepare_tui_data_payload()
            initial_tui_message = TUIWebSocketMessage(
                event_type="TUI_INITIAL_STATE",
                payload=initial_payload,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            await websocket.send_json(initial_tui_message.model_dump())

            while True:
                data = await websocket.receive_json()
                await websocket.send_json({
                    "type": "acknowledgment",
                    "content": "Message received"
                })

        except WebSocketDisconnect:
            self.logger.info(f"WebSocket client disconnected.")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast_state_update(self, event: Event):
        if not self.active_connections:
            return

        try:
            tui_payload = await self._prepare_tui_data_payload()
            tui_websocket_message = TUIWebSocketMessage(
                event_type="TUI_REFRESH_DATA",
                payload=tui_payload,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            message_to_send = tui_websocket_message.model_dump()

            for connection in list(self.active_connections):
                try:
                    await connection.send_json(message_to_send)
                except Exception as e:
                    self.logger.warning(f"Error broadcasting to client: {e}. Removing connection.")
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)
        except Exception as e:
            self.logger.error(f"Failed to broadcast state update: {e}", exc_info=True)

    async def _prepare_tui_data_payload(self) -> TUIDataPayload:
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
            needs=tui_needs if tui_needs else None,
            behavior=tui_behavior,
            emotional_state=tui_emotional_state if tui_emotional_state else None
        )

        raw_conversation_data = self.state_bridge.get_latest_raw_conversation_state()
        if not isinstance(raw_conversation_data, list):
            actual_recent_raw = []
        else:
            actual_recent_raw = raw_conversation_data[-6:]

        current_recent_messages = [TUIMessage(**msg) for msg in actual_recent_raw]
        current_cognitive_summary = self.state_bridge.get_latest_cognitive_summary()

        model_name_from_config = "N/A"
        if hasattr(Config, 'get_cognitive_model'):
            try:
                model_name_from_config = Config.get_cognitive_model() or "N/A"
            except Exception:
                model_name_from_config = "N/A"

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
        uvicorn.run(self.app, host=host, port=port)
