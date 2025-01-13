# modules/behaviors/behavior_manager.py

from .idle import IdleBehavior
from .walk import WalkBehavior
from .chase import ChaseBehavior
from .sleep import SleepBehavior
from .relax import RelaxBehavior
from event_dispatcher import global_event_dispatcher, Event
import time
import random

class BehaviorManager:
    """
    Manages behaviors through a probabilistic state transition system.

    The behavior system models natural activity patterns by:
    {immediate reactions} -> {needs drive behavior changes}
    {energy management} -> {alternates between active and restful states}
    {mood influence} -> {emotional state affects behavior choices}
    {fuzzy determinism} -> {behaviors emerge naturally but not predictably}

    Behaviors affect needs directly through  rate modification,
    creating a natural feedback loop between activity and internal state.
    Future integrations with cognitive systems will allow for more
    deliberate behavior control and learning.
    """

    BEHAVIOR_PATTERNS = {
        'idle': {
            'base_weight': 0.3,
            'transitions': {
                'walk': {
                    'trigger': 'boredom',
                    'probability': lambda needs: 1 - needs['boredom']['satisfaction'],  # Inverse of satisfaction
                    'min_threshold': 0.3  # 30% satisfaction or below to start walk
                },
                'relax': {
                    'trigger': 'stamina',
                    'probability': lambda needs: 1 - needs['stamina']['satisfaction'],
                    'min_threshold': 0.6  # Relax triggers if stamina is less than 60% satisfied
                }
            }
        },
        'walk': {
            'base_weight': 0.2,
            'transitions': {
                'chase': {
                    'trigger': 'boredom',
                    'probability': lambda needs: 1 - needs['boredom']['satisfaction'],
                    'min_threshold': 0.5,
                    'condition': lambda needs: needs['stamina']['satisfaction'] > 0.4  # Only chase if stamina is above 40%
                },
                'relax': {
                    'trigger': 'stamina',
                    'probability': lambda needs: 1 - needs['stamina']['satisfaction'],
                    'min_threshold': 0.3
                },
                'idle': {
                    'trigger': 'boredom',
                    'probability': lambda needs: needs['boredom']['satisfaction'],
                    'min_threshold': 0.3
                }
            }
        },
        'relax': {
            'base_weight': 0.2,
            'transitions': {
                'sleep': {
                    'trigger': 'stamina',
                    'probability': lambda needs: 1 - needs['stamina']['satisfaction'],
                    'min_threshold': 0.2
                },
                'idle': {
                    'trigger': 'stamina',
                    'probability': lambda needs: needs['stamina']['satisfaction'],
                    'min_threshold': 0.6
                }
            }
        },
        'sleep': {
            'base_weight': 0.1,
            'force_threshold': {
                'trigger': 'stamina',
                'value': 0.1  # Trigger sleep when stamina satisfaction drops below 10%
            },
            'transitions': {
                'idle': {
                    'trigger': 'stamina',
                    'probability': lambda needs: needs['stamina']['satisfaction'],
                    'min_threshold': 0.8  # Wake up if stamina is above 80% satisfied
                }
            }
        }
    }
    
    def __init__(self, internal_context, needs_manager):
        """
        Initializes the BehaviorManager.

        Args:
            internal_context (InternalContext): methods to retrieve current internal state
            needs_manager (NeedsManager): The NeedsManager instance (used by behaviors to manage rates)
        """
        self.internal_context = internal_context
        self.needs_manager = needs_manager
        self.current_behavior = None
        self.locked_until = 0
        self.locked_by = None

        self.behaviors = {
            'idle': IdleBehavior(self),
            'walk': WalkBehavior(self),
            'chase': ChaseBehavior(self),
            'sleep': SleepBehavior(self),
            'relax': RelaxBehavior(self)
        }

        self.change_behavior('idle')
        self.setup_event_listeners()

    def setup_event_listeners(self):
        global_event_dispatcher.add_listener("need:changed", self.determine_behavior)
        global_event_dispatcher.add_listener("action:completed", self.determine_behavior)
        global_event_dispatcher.add_listener("mood:changed", self.determine_behavior)
        global_event_dispatcher.add_listener("emotion:new", self.determine_behavior)

        global_event_dispatcher.add_listener("memory:echo", self._handle_memory_echo)

    def _handle_memory_echo(self, event):
        """
        Handle memory echo effects on behavior by creating a light behavioral resonance
        based on remembered states.
        """
        echo_data = event.data
        if not echo_data or 'metadata' not in echo_data:
            return

        # Extract remembered behavior state if present
        remembered_behavior = echo_data['metadata'].get('behavior', {})
        if not remembered_behavior:
            return

        # Calculate echo intensity factoring system-wide echo intensity
        behavior_intensity = echo_data.get('intensity', 0.3) * 0.4  # Reduced impact vs direct needs/emotions
        
        # If behavior matches current, temporarily lock it
        if remembered_behavior.get('name') == self.current_behavior.name:
            # Brief lock to reinforce behavior
            lock_duration = 1.5 * behavior_intensity
            self.locked_until = time.time() + lock_duration
            self.locked_by = 'memory_echo'
        else:
            # Force behavior recheck with current context
            self.determine_behavior(event)

    def update(self):
        """
        Updates the current behavior.
        """
        if self.current_behavior:
            self.current_behavior.update()

    def change_behavior(self, new_behavior_name):
        """
        Changes the current behavior.

        Args:
            new_behavior_name (str): selected behavior to activate
        """
        old_behavior = None  # Ensure old_behavior is always defined

        if self.current_behavior:
            old_behavior = self.current_behavior
            self.current_behavior.stop()
            
        self.current_behavior = self.behaviors[new_behavior_name]
        self.current_behavior.start()

        global_event_dispatcher.dispatch_event_sync(Event("behavior:changed", {
            "old_behavior": old_behavior.name if old_behavior else None,
            "new_behavior": new_behavior_name
        }))

    def is_locked(self):
        """
        Checks if the behavior is currently locked.

        Returns:
            bool: True if the behavior is locked, False otherwise.
        """
        return time.time() < self.locked_until

    def force_behavior(self, behavior_name, duration=None):
        """
        Forces a specific behavior for a given duration.

        Args:
            behavior_name (str): The name of the behavior to force.
            duration (float, optional): The duration in seconds to lock the behavior. If not provided, the behavior will be locked indefinitely.
        """
        if behavior_name in self.behaviors:
            self.change_behavior(behavior_name)
            if duration:
                self.locked_until = time.time() + duration
                self.locked_by = 'forced'
            else:
                self.locked_until = float('inf')
                self.locked_by = 'forced'

    def determine_behavior(self, event):
        """
        Determines the appropriate behavior based on the current state and event.

        Args:
            event (Event): The event that triggered the behavior determination.
        """
        event_type = event.event_type
        event_data = event.data

        current_needs = self.internal_context.get_current_needs()
        current_mood = self.internal_context.get_current_mood()
        recent_emotions = self.internal_context.get_recent_emotions()
        
        new_behavior = self._calculate_behavior(event_type, event_data, current_needs, current_mood, recent_emotions)

        if new_behavior != self.current_behavior.name:
            self.change_behavior(new_behavior)

    def _calculate_behavior(self, event_type, event_data, current_needs, current_mood, recent_emotions):
        """
        Calculates the appropriate behavior based on the current state and event using a probabilistic transition system.

        Args:
            event_type (str): The type of event that triggered the behavior calculation.
            event_data (dict): Additional data associated with the event.
            current_needs (dict): The current state of the internal's needs.
            current_mood (dict): The current mood info of the internal.
            recent_emotions (list): A list of recent EmotionalVector objects.

        Returns:
            str: The name of the selected behavior.
        """

        if self.is_locked():
            print("Behavior is locked; returning current behavior.")
            return self.current_behavior.name
        
        current = self.current_behavior.name
        pattern = self.BEHAVIOR_PATTERNS[current]

        # Step 1: Check force_threshold conditions
        for behavior, data in self.BEHAVIOR_PATTERNS.items():
            if 'force_threshold' in data:
                threshold = data['force_threshold']
                current_value = current_needs[threshold['trigger']]['satisfaction']
                if current_value <= threshold['value']:
                    return behavior

        # Step 2: Calculate transition weights
        transition_weights = {}
        for next_behavior, rules in pattern['transitions'].items():
            current_value = current_needs[rules['trigger']]['satisfaction']
            probability = rules['probability'](current_needs)
            if current_value <= rules['min_threshold']:
                if 'condition' in rules and not rules['condition'](current_needs):
                    continue
                transition_weights[next_behavior] = probability * random.random()

        if not transition_weights:
            # Default to current behavior or choose a safe fallback
            return current

        # Step 3: Add weight for staying in current behavior
        transition_weights[current] = pattern['base_weight'] * random.random()
        
        # Step 4: Determine the highest-weighted behavior
        try:
            selected_behavior = max(transition_weights.items(), key=lambda x: x[1])[0]
            return selected_behavior
        except Exception as e:
            print(f"Error selecting behavior from weights: {e}")
            raise e

    def get_current_behavior(self):
        """
        Returns the current behavior.

        Returns:
            Behavior: The current behavior instance.
        """
        return self.current_behavior

    def get_behavior_state(self):
        """
        Gets the persistent state of the behavior system.
        """
        return {
            "current_behavior": self.current_behavior.name if self.current_behavior else None,
            **({"locked_until": self.locked_until, "locked_by": self.locked_by} 
               if self.is_locked() else {})
        }

    def set_behavior_state(self, state):
        """
        Sets the state of the behavior system.
        """
        if not state:
            return

        # Set lock state first if present
        if "locked_until" in state and "locked_by" in state:
            self.locked_until = state["locked_until"]
            self.locked_by = state["locked_by"]
        else:
            self.locked_until = 0
            self.locked_by = None

        # Change to specified behavior if provided
        if "current_behavior" in state and state["current_behavior"] in self.behaviors:
            self.change_behavior(state["current_behavior"])
        else:
            self.change_behavior("idle")