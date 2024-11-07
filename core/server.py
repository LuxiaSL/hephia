"""
Core server implementation for Hephia.

This server acts as the central nervous system, coordinating communication
between the pet's internal systems, LLM brain, and external interfaces.
It manages both HTTP endpoints for commands and WebSocket connections
for real-time state updates.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any
import asyncio
import time
import os
from datetime import datetime
from pydantic import BaseModel

from core.timer import TimerCoordinator
from core.state_bridge import StateBridge
from event_dispatcher import global_event_dispatcher, Event
from pet.pet import Pet
from config import Config
from brain.environments.ui_formatter import UIFormatter
from brain.command_preprocessor import CommandPreprocessor
from brain.environments.environment_registry import EnvironmentRegistry
from brain.logging_utils import ExoLogger
from api_clients import APIManager

class CommandRequest(BaseModel):
    """Model for incoming commands."""
    messages: List[Dict[str, str]]
    stream: bool = False

class StateUpdateResponse(BaseModel):
    """Model for state updates."""
    event_type: str
    data: Dict[str, Any]

class HephiaServer:
    """
    Main server class coordinating all Hephia's systems.
    """
    
    def __init__(self):
        """Initialize the server and its components."""
        self.app = FastAPI(title="Hephia Server")
        self.setup_middleware()
        self.active_connections: List[WebSocket] = []

        # Add lock for exo loop
        self.exo_lock = asyncio.Lock()
        self.last_successful_exo = None  # Track last successful completion

        self.api = APIManager.from_env()

        self.pet = Pet()
        self.state_bridge = StateBridge(pet=self.pet)
        self.timer = TimerCoordinator()
        
        self.command_preprocessor = CommandPreprocessor(self.api)
        self.environment_registry = EnvironmentRegistry(self.api)
        self.exo_messages = []  # Store conversation context

        self.setup_routes()
        self.setup_event_handlers()
    
    def setup_middleware(self):
        """Configure server middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Adjust for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
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
        async def chat_completions(request: CommandRequest):
            """Handle LLM interactions and commands."""
            try:
                response = await self.state_bridge.process_command(request.messages)
                return {
                    "id": f"chat-{id(response)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "hephia-1",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response
                        },
                        "finish_reason": "stop"
                    }]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/state")
        async def get_state():
            """Get current system state."""
            return await self.state_bridge.get_current_state()
    
    def setup_event_handlers(self):
        """Set up handlers for system events."""
        # Create tasks for async handlers
        global_event_dispatcher.add_listener(
            "state:changed",
            lambda event: asyncio.create_task(self.broadcast_state_update(event))
        )
        global_event_dispatcher.add_listener(
            "pet:action",
            lambda event: asyncio.create_task(self.broadcast_state_update(event))
        )
        global_event_dispatcher.add_listener(
            "timer:task_executed",
            lambda event: asyncio.create_task(self.handle_timer_event(event))
        )
    
    async def startup(self):
        """Initialize all systems in correct order."""
        # Start pet systems
        await self.pet.start()
        
        # Initialize bridge with existing pet
        await self.state_bridge.initialize()
        
        # Configure timer tasks
        self.timer.add_task(
            name="needs_update",
            interval=Config.NEED_UPDATE_TIMER,
            callback=self.pet.update_needs
        )
        
        self.timer.add_task(
            name="emotions_update",
            interval=Config.EMOTION_UPDATE_TIMER,
            callback=self.pet.update_emotions
        )

        # Add exo loop timer task
        self.timer.add_task(
            name="exo_update",
            interval=Config.EXO_LOOP_TIMER,
            callback=self.process_exo_loop
        )
        
        # Start timer
        asyncio.create_task(self.timer.run())

    async def process_exo_loop(self):
        """Process periodic exo/LLM interactions with command preprocessing."""
        # Check if we should run based on last successful completion
        if self.last_successful_exo and \
           (datetime.now() - self.last_successful_exo).total_seconds() < Config.EXO_MIN_INTERVAL:
            print("Skipping exo loop - too soon since last completion")
            return
        try:
            async with asyncio.timeout(Config.EXO_TIMEOUT):
                if self.exo_lock.locked():
                    print("Exo loop already in progress, skipping")
                    return
                async with self.exo_lock:
                    try:
                        # Get current state
                        state = await self.state_bridge.get_current_state()
                        ExoLogger.log_state_update(state)  # Log state
                        formatted_state = UIFormatter.format_terminal_view(state)
                        
                        # Maintain conversation context
                        self.exo_messages.append({
                            "role": "user",
                            "content": f"Current state:\n{formatted_state} WELCOME TO YOUR NEW OPERATING SYSTEM. only enter 'help' to begin, then explore your available commands."
                        })
                        
                        # trim after a while
                        if len(self.exo_messages) > Config.EXO_MAX_MESSAGES: 
                            self.exo_messages = self.exo_messages[-Config.EXO_MAX_MESSAGES:]
                        
                        # Get LLM response
                        async with asyncio.timeout(Config.LLM_TIMEOUT):
                            response = await self.api.openpipe.create_completion(
                                model=os.getenv("OPENPIPE_MODEL"),
                                messages=[
                                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                                    *self.exo_messages
                                ],
                                temperature=Config.EXO_TEMPERATURE,
                            )
                        
                        llm_response = response["choices"][0]["message"]["content"]
                        ExoLogger.log_llm_exchange(self.exo_messages, llm_response)
                        
                        self.exo_messages.append({
                            "role": "assistant",
                            "content": llm_response
                        })
                        
                        # Preprocess command
                        command, help_text = await self.command_preprocessor.preprocess_command(
                            llm_response,
                            self.environment_registry.get_available_commands()
                        )
                        
                        ExoLogger.log_command_processing(llm_response, command, help_text)

                        if command:
                            # Process valid command
                            await self.state_bridge.process_command([{
                                "role": "assistant",
                                "content": command
                            }])
                        elif help_text == "Too many command errors - resetting conversation":
                            # Reset conversation on too many errors
                            self.exo_messages = []
                            print("Exo loop conversation reset due to repeated errors")
                        else:
                            print(f"Command processing message: {help_text}")

                        # Update last successful completion time
                        self.last_successful_exo = datetime.now()
                        
                    except asyncio.TimeoutError:
                        print("Timeout in LLM request")
                        self.exo_messages = self.exo_messages[:-1] 
                    except Exception as e:
                        print(f"Error in exo loop: {e}")
                        self.exo_messages = self.exo_messages[:-1]
        except asyncio.TimeoutError:
            print("Timeout waiting for exo lock")
        except Exception as e:
            print(f"Fatal error in exo loop: {e}")
    
    async def shutdown(self):
        """Clean shutdown of all systems."""
        self.timer.stop()
        self.pet.stop()
        await self.state_bridge.save_state()

    
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
            initial_state = await self.state_bridge.get_current_state()
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
        
        Args:
            websocket: The WebSocket connection
            message: The message to handle
        """
        # Process different message types
        message_type = message.get("type")
        if message_type == "command":
            response = await self.state_bridge.process_command(message["data"])
            await websocket.send_json({
                "type": "command_response",
                "data": response
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
    
    async def handle_timer_event(self, event: Event):
        """
        Handle timer-triggered events.
        
        Args:
            event: The timer event to handle
        """
        await self.state_bridge.process_timer_event(event)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the server.
        
        Args:
            host: Host to run on
            port: Port to run on
        """
        uvicorn.run(self.app, host=host, port=port)