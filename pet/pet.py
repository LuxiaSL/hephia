"""
Pet core implementation.
Modified to work with server-based architecture.
"""

from typing import List, Optional
import asyncio

from pet.pet_context import PetContext
from pet.modules.needs.needs_manager import NeedsManager
from pet.modules.behaviors.behavior_manager import BehaviorManager
from pet.modules.actions.action_manager import ActionManager
from pet.modules.emotions.emotional_processor import EmotionalProcessor
from pet.modules.cognition.cognitive_bridge import CognitiveBridge
from pet.modules.emotions.mood_synthesizer import MoodSynthesizer
from pet.modules.memory.memory_system import MemorySystem
from event_dispatcher import global_event_dispatcher, Event

class Pet:
    """
    Core pet class, now designed to work with server architecture.
    """

    def __init__(self):
        """Initialize pet systems."""
        # Initialize context first
        self.context = PetContext(self)
        
        # Initialize managers
        self.needs_manager = NeedsManager()
        self.behavior_manager = BehaviorManager(self.context, self.needs_manager)
        self.action_manager = ActionManager(self.needs_manager)

        # Initialize cognitive and emotional modules
        self.memory_system = MemorySystem()
        self.mood_synthesizer = MoodSynthesizer(self.context)
        self.cognitive_bridge = CognitiveBridge()
        self.emotional_processor = EmotionalProcessor(
            self.context,
            self.cognitive_bridge,
            self.memory_system
        )

        # Set up event listeners
        self.setup_event_listeners()
        
        # Initialize state
        self.is_active = False
        self._update_tasks: List[asyncio.Task] = []

    def setup_event_listeners(self):
        """Set up event listeners for pet systems."""
        global_event_dispatcher.add_listener("need:changed", self.on_need_change)
        global_event_dispatcher.add_listener("behavior:changed", self.on_behavior_change)
        global_event_dispatcher.add_listener("action:performed", self.on_action_performed)
        global_event_dispatcher.add_listener("mood:changed", self.on_mood_change)
        global_event_dispatcher.add_listener("emotion:new", self.on_new_emotion)

    async def start(self):
        """
        Start pet systems.
        Now handled by server's timer coordinator instead of internal timer.
        """
        self.is_active = True
        # Notify system that pet is ready
        global_event_dispatcher.dispatch_event(Event("pet:started", None))

    def stop(self):
        """Stop pet systems."""
        self.is_active = False
        if self.behavior_manager.current_behavior:
            self.behavior_manager.current_behavior.stop()
        
        # Notify system of shutdown
        global_event_dispatcher.dispatch_event(Event("pet:stopped", None))

    async def update_needs(self):
        """Update needs - now called by timer coordinator."""
        if self.is_active:
            self.needs_manager.update_needs()

    async def update_emotions(self):
        """Update emotions - now called by timer coordinator."""
        if self.is_active:
            self.emotional_processor.update()

    def on_need_change(self, event):
        """Handle need change events."""
        if not self.is_active:
            return
        print(f"Pet: Need '{event.data['need_name']}' changed to {event.data['new_value']}")

    def on_behavior_change(self, event):
        """Handle behavior change events."""
        if not self.is_active:
            return
        print(f"Pet: Behavior changed to {event.data['new_behavior']}")

    def on_action_performed(self, event):
        """Handle action performed events."""
        if not self.is_active:
            return
        print(f"Pet: Action '{event.data['action_name']}' performed")

    def on_mood_change(self, event):
        """Handle mood change events."""
        if not self.is_active:
            return
        print(f"Pet: Mood changed to {event.data.get('new_name')}")

    def on_new_emotion(self, event):
        """Handle new emotion events."""
        if not self.is_active:
            return
        print(f"Pet: Emotion experienced: {event.data.get('emotion').name}")

    def perform_action(self, action_name: str):
        """Perform a pet action."""
        if self.is_active:
            self.action_manager.perform_action(action_name)