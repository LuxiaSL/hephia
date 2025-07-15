"""
brain/core/processor.py

Core orchestration layer that manages all cognitive components and interfaces.
Handles initialization, lifecycle management, and coordination between
all system components while maintaining proper separation of concerns.
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from brain.interfaces.base import CognitiveInterface, NotificationManager
from brain.interfaces.exo import ExoProcessorInterface
from brain.interfaces.discord import DiscordInterface
from brain.interfaces.user import UserInterface
from brain.interfaces.action import ActionInterface
from brain.core.command_handler import CommandHandler
from brain.cognition.notification import Notification, NotificationManager
from brain.cognition.memory.manager import MemoryManager
from brain.environments.environment_registry import EnvironmentRegistry
from brain.interfaces.exo_utils.hud.construct import HudConstructor
from core.state_bridge import StateBridge
from core.discord_service import DiscordService
from internal.modules.cognition.cognitive_bridge import CognitiveBridge
from api_clients import APIManager
from loggers import BrainLogger
from config import Config
from event_dispatcher import global_event_dispatcher, Event

class CoreProcessor:
    def __init__(
        self,
        api_manager: APIManager,
        state_bridge: StateBridge,
        cognitive_bridge: CognitiveBridge,
        environment_registry: EnvironmentRegistry,
        discord_service: DiscordService
    ):
        self.api = api_manager
        self.state_bridge = state_bridge
        self.cognitive_bridge = cognitive_bridge
        self.environment_registry = environment_registry
        self.discord_service = discord_service
        
        # Core components (initialized in setup)
        self.command_handler: Optional[CommandHandler] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.notification_manager: Optional[NotificationManager] = None
        self.hud_constructor: Optional[HudConstructor] = None
        
        # Interfaces
        self.interfaces: Dict[str, CognitiveInterface] = {}
        
        # State tracking
        self.is_running = False
        self.last_error: Optional[Exception] = None
        self._setup_lock = asyncio.Lock()
        
        # Configuration
        self.initialization_timeout = timedelta(seconds=Config.INITIALIZATION_TIMEOUT)
        self.shutdown_timeout = timedelta(seconds=Config.SHUTDOWN_TIMEOUT)

    async def initialize(self) -> None:
        """
        Initialize all system components in proper order.
        Ensures dependencies are set up correctly.
        """
        try:
            async with asyncio.timeout(self.initialization_timeout.total_seconds()):
                async with self._setup_lock:
                    BrainLogger.info("Starting system initialization")
                    
                    # 1. Initialize core components
                    await self._initialize_core_components()
                    
                    # 2. Initialize interfaces
                    await self._initialize_interfaces()
                    
                    # 3. Setup event listeners
                    self._setup_event_listeners()
                    
                    # 4. Load initial state
                    await self._load_initial_state()
                    
                    BrainLogger.info("System initialization complete")
                    
        except asyncio.TimeoutError:
            error_msg = "System initialization timed out"
            BrainLogger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            BrainLogger.error(f"System initialization failed: {e}")
            raise

    async def _initialize_core_components(self) -> None:
        """Initialize core system components in dependency order."""
        try:
            self.notification_manager = NotificationManager()
            
            self.command_handler = CommandHandler(
                self.api,
                self.environment_registry,
                self.state_bridge
            )
            
            self.memory_manager = MemoryManager(
                self.state_bridge,
                self.cognitive_bridge,
                self.api,
                {}
            )

            self.hud_constructor = HudConstructor(self.discord_service)

        except Exception as e:
            BrainLogger.error(f"Core component initialization failed: {e}")
            raise

    async def _initialize_interfaces(self) -> None:
        """Initialize and register all interfaces."""
        try:
            # Initialize interfaces
            self.interfaces['exo_processor'] = ExoProcessorInterface(
                self.api,
                self.cognitive_bridge,
                self.state_bridge,
                self.command_handler,
                self.notification_manager,
                self.hud_constructor
            )
            
            self.interfaces['discord'] = DiscordInterface(
                self.api,
                self.state_bridge,
                self.cognitive_bridge,
                self.notification_manager
            )
            
            self.interfaces['user'] = UserInterface(
                self.api,
                self.state_bridge,
                self.cognitive_bridge,
                self.notification_manager
            )

            self.interfaces['action'] = ActionInterface(
                self.state_bridge,
                self.cognitive_bridge,
                self.notification_manager,
                self.api
            )
            
            # Update memory manager with interfaces
            self.memory_manager = MemoryManager(
                self.state_bridge,
                self.cognitive_bridge,
                self.api,
                self.interfaces
            )
            
        except Exception as e:
            BrainLogger.error(f"Interface initialization failed: {e}")
            raise

    def _setup_event_listeners(self) -> None:
        """Setup global event listeners for system coordination."""
        # System lifecycle events
        global_event_dispatcher.add_listener(
            "system:error",
            self._handle_system_error, 
            priority=10  # Higher priority for error handling
        )
        
        # Error events
        global_event_dispatcher.add_listener(
            "system:component_error",
            self._handle_component_error,
            priority=10  # Higher priority for error handling
        )

        global_event_dispatcher.add_listener(
            "memory:conflict_detected",
            self._handle_memory_conflict
        )

    async def _load_initial_state(self) -> None:
        """Load initial system state and perform startup validation."""
        try:
            # Load persistent state
            brain_state = self.state_bridge.persistent_state.brain_state if self.state_bridge.persistent_state else None
            
            # Ensure brain_state is properly formatted for conversation history
            initial_state = brain_state if brain_state and isinstance(brain_state, list) else []
                
            # Initialize ExoProcessor conversation with validated state
            await self.interfaces['exo_processor'].initialize(initial_state)
            
            # Validate system readiness
            await self._validate_system_state()
            
        except Exception as e:
            BrainLogger.error(f"Initial state loading failed: {e}")
            raise

    async def _validate_system_state(self) -> None:
        """Validate all components are properly initialized and responsive."""
        try:
            # Check core components
            if not all([
                self.command_handler,
                self.memory_manager,
                self.notification_manager,
                all(isinstance(i, CognitiveInterface) for i in self.interfaces.values())
            ]):
                raise RuntimeError("Core components not fully initialized")
                
        except Exception as e:
            BrainLogger.error(f"System validation failed: {e}")
            raise

    async def start(self) -> None:
        """Start the system and begin processing."""
        try:
            if self.is_running:
                return
                
            self.is_running = True
            BrainLogger.info("Starting system processing")
            
            # Start main processing loop
            asyncio.create_task(self._process_loop())
            
        except Exception as e:
            self.is_running = False
            BrainLogger.error(f"System start failed: {e}")
            raise

    async def stop(self) -> None:
        """
        Gracefully stop system processing.
        Ensures all components shut down properly.
        """
        try:
            if not self.is_running:
                return
                
            async with asyncio.timeout(self.shutdown_timeout.total_seconds()):
                self.is_running = False
                BrainLogger.info("Initiating system shutdown")
                
                # Clean up components
                await self._cleanup_components()
                
                BrainLogger.info("System shutdown complete")
                
        except asyncio.TimeoutError:
            BrainLogger.error("System shutdown timed out")
        except Exception as e:
            BrainLogger.error(f"System shutdown error: {e}")
            raise

    async def _process_loop(self) -> None:
        """Main processing loop coordinating all components."""
        while self.is_running:
            try:
                # Process ExoProcessor turn
                await self.interfaces['exo_processor'].process_interaction(None)
                
                # Periodic maintenance
                await self._perform_maintenance()
                
                # Respect rate limiting
                await asyncio.sleep(Config.get_exo_min_interval())
                
            except Exception as e:
                BrainLogger.error(f"Processing loop error: {e}")
                self.last_error = e
                await self._handle_system_error(Event(
                    "system:error",
                    {"error": e, "source": "process_loop"}
                ))

    async def _perform_maintenance(self) -> None:
        """Perform periodic system maintenance tasks."""
        try:
            # Check component health
            await self._validate_system_state()
            
        except Exception as e:
            BrainLogger.error(f"Maintenance error: {e}")

    async def _cleanup_components(self) -> None:
        """Clean up and close all components."""
        try:
            # Clean up interfaces
            for interface in self.interfaces.values():
                try:
                    # Assuming interfaces might need cleanup
                    if hasattr(interface, 'cleanup'):
                        await interface.cleanup()
                except Exception as e:
                    BrainLogger.error(f"Interface cleanup error: {e}")
            
        except Exception as e:
            BrainLogger.error(f"Component cleanup failed: {e}")

    async def _handle_system_error(self, event: Event) -> None:
        """Handle system-level errors."""
        error = event.data.get('error')
        source = event.data.get('source', 'unknown')
        
        BrainLogger.error(f"System error from {source}: {error}")
        
        if self.is_running:
            # Attempt recovery based on error type
            if isinstance(error, (asyncio.TimeoutError, ConnectionError)):
                # Connection issues - wait and retry
                await asyncio.sleep(5)
            elif isinstance(error, RuntimeError):
                # Serious runtime error - consider restart
                await self.stop()
                await self.initialize()
                await self.start()
            else:
                # Unknown error - log and continue
                BrainLogger.error(f"Unhandled error type: {type(error)}")

    async def _handle_component_error(self, event: Event) -> None:
        """Handle component-specific errors."""
        component = event.data.get('component')
        error = event.data.get('error')
        
        BrainLogger.error(f"Component {component} error: {error}")
        
        # Attempt component-specific recovery
        if component in self.interfaces:
            try:
                # Reinitialize interface
                await self._initialize_interfaces()
            except Exception as e:
                BrainLogger.error(f"Interface recovery failed: {e}")

    async def handle_chat_completion(
        self,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Handle chat completion requests in OpenAI format."""
        try:
            # Process through user interface - it handles its own notifications
            response_text = await self.interfaces['user'].process_interaction({
                'messages': messages
            })

            # Return OpenAI-style response
            return {
                "id": f"chat-{id(response_text)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": 'hephia',
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }]
            }

        except Exception as e:
            BrainLogger.error(f"Chat completion error: {e}")
            raise

    async def handle_discord_message(
        self,
        message_context: Dict[str, Any]
    ) -> str:
        """Handle incoming Discord messages."""
        try:
            # Ensure channel path is included in the message context
            channel_data = message_context.get('channel', {})
            if 'path' not in channel_data and 'name' in channel_data:
                channel_name = channel_data.get('name', 'Unknown')
                guild_name = channel_data.get('guild_name')
                channel_data['path'] = f"{guild_name}:{channel_name}" if guild_name else channel_name
                message_context['channel'] = channel_data

            return await self.interfaces['discord'].process_interaction(
                message_context
            )

        except Exception as e:
            BrainLogger.error(f"Discord message error: {e}")
            raise

    async def handle_discord_channel_update(
        self,
        channel_id: str,
        channel_name: str,
        guild_name: Optional[str] = None,
        message_count: int = 0
    ) -> None:
        """Handle Discord channel activity updates."""
        try:
            # Create channel path for user-friendly reference
            channel_path = f"{guild_name}:{channel_name}" if guild_name else channel_name
            
            # Create notification about channel activity using proper structure
            notification = Notification(
                content={
                    "channel_name": channel_name,
                    "guild_name": guild_name, 
                    "path": channel_path,  # Add path-based reference
                    "message_count": message_count,
                    "update_type": "channel_activity"
                },
                source_interface="discord",
                timestamp=datetime.now()
            )
            await self.notification_manager.add_notification(notification)

        except Exception as e:
            BrainLogger.error(f"Discord channel update error: {e}")
            raise

    async def handle_action_request(
        self,
        content: Dict[str, Any]
    ) -> Notification:
        """Handle action requests from other components."""
        try:
            # Process through Action interface - it handles its own notifications
            return await self.interfaces['action'].create_notification(content)

        except Exception as e:
            BrainLogger.error(f"Action request error: {e}")
            raise e
        
    async def _handle_memory_conflict(self, event: Event) -> None:
        """Handle memory conflicts detected by the MemoryManager."""
        try:
            conflict_data = event.data.get('conflict_data')
            if not conflict_data:
                BrainLogger.warning("Memory conflict event missing data")
                return

            self.memory_manager.handle_conflict(
                node_a_id=conflict_data.get('node_a_id'),
                node_b_id=conflict_data.get('node_b_id'),
                conflicts=conflict_data.get('conflicts'),
                metrics=conflict_data.get('metrics')
            )

        except Exception as e:
            BrainLogger.error(f"Memory conflict handling error: {e}")
            raise e

    async def prune_conversation(self) -> None:
        """Prune conversation history to maintain system performance."""
        try:
            # Prune conversation through ExoProcessor interface
            await self.interfaces['exo_processor'].prune_conversation()
        except Exception as e:
            BrainLogger.error(f"Conversation pruning error: {e}")
            raise e