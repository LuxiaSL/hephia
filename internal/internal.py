"""
Internal core implementation.
Modified to work with server-based architecture.
"""

from __future__ import annotations
from typing import List
import asyncio

from internal.internal_context import InternalContext
from internal.modules.needs.needs_manager import NeedsManager
from internal.modules.behaviors.behavior_manager import BehaviorManager
from internal.modules.actions.action_manager import ActionManager
from internal.modules.emotions.emotional_processor import EmotionalProcessor
from internal.modules.cognition.cognitive_bridge import CognitiveBridge
from internal.modules.emotions.mood_synthesizer import MoodSynthesizer
from internal.modules.memory.memory_system import MemorySystemOrchestrator

from config import Config
from event_dispatcher import global_event_dispatcher, Event
from loggers import InternalLogger, MemoryLogger  


class Internal:
    """
    Core internal class, now designed to work with server architecture.
    """

    def __init__(self, api_manager) -> None:
        """Synchronous initialization of core internal systems."""
        # Initialize context & state control
        self.context = InternalContext(self)

        # Initialize managers that don't require async work
        self.needs_manager = NeedsManager()
        self.behavior_manager = BehaviorManager(self.context, self.needs_manager)
        self.action_manager = ActionManager(self.needs_manager)

        # Placeholders for async-initialized modules
        self.memory_system = None  # type: MemorySystemOrchestrator
        self.mood_synthesizer = None  # type: MoodSynthesizer
        self.cognitive_bridge = None  # type: CognitiveBridge
        self.emotional_processor = None  # type: EmotionalProcessor

        # Set up event listeners
        self.setup_event_listeners()

        # Initialize state and update tasks container
        self.is_active = False
        self._update_tasks: List[asyncio.Task] = []

    @classmethod
    async def create(cls, api_manager) -> Internal:
        """
        Asynchronously creates an instance of Internal.
        
        This factory method performs all async initialization (like awaiting the memory system)
        and returns a fully initialized instance.
        """
        instance = cls(api_manager)

        # Await async initialization for memory_system
        instance.memory_system = await MemorySystemOrchestrator.create(api_manager, instance.context, update_interval=Config.MEMORY_UPDATE_TIMER)

        # Now initialize the modules that depend on memory_system and context.
        instance.mood_synthesizer = MoodSynthesizer(instance.context)
        instance.cognitive_bridge = CognitiveBridge(instance.context, instance.memory_system)
        instance.emotional_processor = EmotionalProcessor(instance.context, instance.cognitive_bridge)

        return instance

    def setup_event_listeners(self):
        """Set up event listeners for internal systems."""
        # silence !
        #global_event_dispatcher.add_listener("need:changed", self.on_need_change)
        global_event_dispatcher.add_listener("behavior:changed", self.on_behavior_change)
        global_event_dispatcher.add_listener("action:performed", self.on_action_performed)
        global_event_dispatcher.add_listener("mood:changed", self.on_mood_change)
        global_event_dispatcher.add_listener("emotion:new", self.on_new_emotion)
        global_event_dispatcher.add_listener("memory:node_created", self.on_node_created)
        #global_event_dispatcher.add_listener("cognitive:context_update", self.on_cognitive_update)

    async def restore_state(self, state_data: dict):
        """
        Restore internal state from persistence data.
        This happens before start() during initialization.
        """
        # Set state in each manager
        self.needs_manager.set_needs_state(state_data.get('needs', {}))
        self.behavior_manager.set_behavior_state(state_data.get('behavior', {}))
        self.emotional_processor.set_emotional_state(state_data.get('emotions', {}))
        self.mood_synthesizer.set_mood_state(state_data.get('mood', {}))

    async def shake(self):
        """
        perform a shake to ensure all state data is propagated
        """
        # First update needs which will trigger need events
        await self.update_needs()
        
        # Process emotional state which will absorb need events
        await self.update_emotions()

        # and now memories
        await self.update_memories()
        
        # Force behavior system to reevaluate with new need states
        current_behavior = self.behavior_manager.current_behavior.name
        await self.behavior_manager.determine_behavior(Event("shake", {
            "current_needs": self.context.get_current_needs(),
            "current_mood": self.context.get_current_mood(),
            "recent_emotions": await self.context.get_recent_emotions()
        }))
        
        # Let one event processing cycle complete
        await asyncio.sleep(0.5)
        
        # Verify state consistency
        if self.behavior_manager.current_behavior.name != current_behavior:
            # Log that shake caused behavior change
            print(f"Shake caused behavior change: {current_behavior} -> {self.behavior_manager.current_behavior.name}")

    async def start(self):
        """Start internal systems."""
        self.is_active = True
        global_event_dispatcher.dispatch_event(Event("internal:started", None))

    async def stop(self):
        """Stop internal systems."""
        self.is_active = False
        if self.behavior_manager.current_behavior:
            self.behavior_manager.current_behavior.stop()

        await self.memory_system.shutdown()
        
        # Notify system of shutdown
        global_event_dispatcher.dispatch_event(Event("internal:stopped", None))

    async def update_needs(self):
        if self.is_active:
            self.needs_manager.update_needs()

    async def update_emotions(self):
        if self.is_active:
            self.emotional_processor.update()
    
    async def update_memories(self):
        if self.is_active:
            await self.memory_system._run_periodic_updates()

    def on_need_change(self, event):
        """Handle need change events."""
        if not self.is_active:
            return
        InternalLogger.log_state_change(
            f"need '{event.data['need_name']}'",
            event.data.get('old_value', 'unknown'),
            event.data['new_value']
        )

    def on_behavior_change(self, event):
        """Handle behavior change events."""
        if not self.is_active:
            return
        InternalLogger.log_state_change(
            'behavior',
            event.data.get('old_name', 'unknown'),
            event.data['new_name']
        )

    def on_action_performed(self, event):
        """Handle action performed events."""
        if not self.is_active:
            return
        InternalLogger.log_behavior(
            event.data['action_name'],
            event.data.get('context')
        )

    def on_mood_change(self, event):
        """Handle mood change events."""
        if not self.is_active:
            return
        InternalLogger.log_state_change(
            'mood',
            event.data.get('old_name', 'unknown'),
            event.data['new_name']
        )

    def on_new_emotion(self, event):
        """Handle new emotion events."""
        if not self.is_active:
            return
        emotion = event.data.get('emotion')
        InternalLogger.log_state_change(
            'emotion',
            'none',
            f"{emotion.name} (v:{emotion.valence:.2f}, a:{emotion.arousal:.2f})"
        )

    def on_node_created(self, event):
        """Handle new memory node events."""
        if not self.is_active:
            return

        node_id = event.data['node_id']
        node_type = event.data['node_type']

        if node_type == 'cognitive':
            text_content = event.data['content']
            # For cognitive nodes, log both ID and content
            MemoryLogger.info(f"Cognitive memory node created: {node_id}")
            MemoryLogger.info(f"Content: {text_content}")
        else:
            # For body nodes, just log the ID 
            MemoryLogger.info(f"Body memory node created: {node_id}")

    def on_cognitive_update(self, event):
        if not self.is_active:
            return
        #print(f"Received cognitive update event: {event.data}")

    def perform_action(self, action_name: str):
        """Perform a internal action."""
        if self.is_active:
            self.action_manager.perform_action(action_name)