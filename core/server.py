"""
Core server implementation for Hephia (Pet).

Slim orchestrator — initializes systems, registers routes via include_router,
and manages lifecycle. Route handlers live in core/routes/, WebSocket handlers
in core/websockets/.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.timer import TimerCoordinator
from core.state_bridge import StateBridge
from core.event_bridge import EventBridge
from core.routes import setup_routes
from core.websockets.tui_stream import TUIStreamManager
from event_dispatcher import global_event_dispatcher, Event
from internal.internal import Internal
from mind.mind import Mind
from mind.introspection import IntrospectionRunner
from config import Config
from api_clients import APIManager
from loggers import SystemLogger


class HephiaServer:
    """
    Pet server orchestrator.
    Initializes internal state, mind, and memory systems.
    Delegates HTTP routes to core/routes/ and WebSocket to core/websockets/.
    """

    def __init__(self) -> None:
        self.app = FastAPI(title="Hephia Pet Server")
        self._setup_middleware()
        self.timer = TimerCoordinator()
        self.api = APIManager.from_env()
        self.logger = SystemLogger

        # Async components (set in create())
        self.internal: Optional[Internal] = None
        self.mind: Optional[Mind] = None
        self.introspection: Optional[IntrospectionRunner] = None
        self.state_bridge: Optional[StateBridge] = None
        self.event_bridge: Optional[EventBridge] = None
        self.tui_stream: Optional[TUIStreamManager] = None

    @classmethod
    async def create(cls) -> HephiaServer:
        """Asynchronously create and initialize a HephiaServer instance."""
        instance = cls()

        # Initialize core systems
        instance.internal = await Internal.create(instance.api)
        instance.mind = Mind(
            bridge=instance.internal.cognitive_bridge,
            api_manager=instance.api,
        )
        instance.introspection = IntrospectionRunner(
            bridge=instance.internal.cognitive_bridge,
            conversation=instance.mind.conversation,
        )
        instance.state_bridge = StateBridge(internal=instance.internal)
        instance.event_bridge = EventBridge(state_bridge=instance.state_bridge)
        instance.tui_stream = TUIStreamManager(instance.state_bridge)

        # Stash on app.state for route dependency injection
        instance.app.state.mind = instance.mind
        instance.app.state.internal = instance.internal
        instance.app.state.state_bridge = instance.state_bridge
        instance.app.state.bridge = instance.internal.cognitive_bridge

        # Services (notes, worker queue, settings) are created lazily in startup
        # or eagerly here if they have no async init requirements
        from services.notes_service import NotesService
        instance.app.state.notes = NotesService()

        from services.worker_queue import WorkerTaskQueue
        instance.app.state.worker_queue = WorkerTaskQueue(instance.mind.worker_model)

        from services.settings_service import SettingsService
        instance.app.state.settings_service = SettingsService(
            bridge=instance.internal.cognitive_bridge,
        )

        # Register routes and WebSocket endpoints
        setup_routes(instance.app)
        instance._setup_websocket_routes()
        instance._setup_lifecycle()
        instance._setup_event_handlers()

        return instance

    def _setup_middleware(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_websocket_routes(self) -> None:
        @self.app.websocket("/ws")
        async def websocket_tui(websocket: WebSocket):
            await self.tui_stream.handle_connection(websocket)

        from core.websockets.chat_stream import ChatStreamManager
        self._chat_stream = ChatStreamManager(self.mind)

        @self.app.websocket("/ws/chat")
        async def websocket_chat(websocket: WebSocket):
            await self._chat_stream.handle_connection(websocket)

    def _setup_lifecycle(self) -> None:
        @self.app.on_event("startup")
        async def startup():
            await self._startup()

        @self.app.on_event("shutdown")
        async def shutdown():
            await self._shutdown()

    def _setup_event_handlers(self) -> None:
        global_event_dispatcher.add_listener(
            "state:changed",
            lambda event: asyncio.create_task(self.tui_stream.broadcast_state_update()),
        )
        global_event_dispatcher.add_listener(
            "internal:action",
            lambda event: asyncio.create_task(self.tui_stream.broadcast_state_update()),
        )

    async def _startup(self) -> None:
        try:
            SystemLogger.info("Initializing state bridge...")
            await self.state_bridge.initialize()

            # Restore conversation state from persisted brain_state
            if self.mind and self.state_bridge.persistent_state:
                saved_conversation = self.state_bridge.persistent_state.brain_state
                if isinstance(saved_conversation, list) and saved_conversation:
                    self.mind.restore_conversation_state(saved_conversation)
                    SystemLogger.info(f"Restored {len(saved_conversation)} conversation messages")

            await self.internal.start()

            SystemLogger.info("Adding internal timers...")
            self.timer.add_task(
                name="needs_update",
                interval=Config.NEED_UPDATE_TIMER,
                callback=self.internal.update_needs,
            )
            self.timer.add_task(
                name="emotions_update",
                interval=Config.EMOTION_UPDATE_TIMER,
                callback=self.internal.update_emotions,
            )

            self.timer.add_task(
                name="memory_maintenance",
                interval=Config.MEMORY_UPDATE_TIMER,  # 180s
                callback=self.internal.update_memories,
            )
            self.timer.add_task(
                name="introspection",
                interval=600,  # every 10 minutes
                callback=self.introspection.maybe_introspect,
            )

            # Worker queue cleanup timer
            self.timer.add_task(
                name="worker_cleanup",
                interval=300,  # every 5 minutes
                callback=self.app.state.worker_queue.cleanup,
            )

            SystemLogger.info("Starting timer...")
            asyncio.create_task(self.timer.run())
        except Exception as e:
            await self._shutdown()
            raise RuntimeError(f"Startup failed: {e}") from e

    async def _shutdown(self) -> None:
        try:
            self.timer.stop()
            await self.internal.stop()
            await self.state_bridge._save_session()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    def run(self, host: str = "0.0.0.0", port: int = 5517) -> None:
        uvicorn.run(self.app, host=host, port=port)
