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
    Manages the pet's behaviors through a probabilistic state transition system.

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
                    'probability': lambda needs: needs['boredom'] / 100.0,
                    'min_threshold': 30
                },
                'relax': {
                    'trigger': 'stamina',
                    'probability': lambda needs: (100 - needs['stamina']) / 100.0,
                    'min_threshold': 40
                }
            }
        },
        'walk': {
            'base_weight': 0.2,
            'transitions': {
                'chase': {
                    'trigger': 'boredom',
                    'probability': lambda needs: (needs['boredom'] - 50) / 50.0,
                    'min_threshold': 50,
                    'condition': lambda needs: needs['stamina'] > 40
                },
                'relax': {
                    'trigger': 'stamina',
                    'probability': lambda needs: max(0, (70 - needs['stamina']) / 70.0),
                    'min_threshold': 30
                },
                'idle': {
                    'trigger': 'boredom',
                    'probability': lambda needs: (30 - needs['boredom']) / 30.0,
                    'min_threshold': 0
                }
            }
        },
        'chase': {
            'base_weight': 0.1,
            'transitions': {
                'relax': {
                    'trigger': 'stamina',
                    'probability': lambda needs: max(0, (60 - needs['stamina']) / 60.0),
                    'min_threshold': 40
                },
                'walk': {
                    'trigger': 'stamina',
                    'probability': lambda needs: max(0, (40 - needs['stamina']) / 40.0),
                    'min_threshold': 20
                }
            }
        },
        'relax': {
            'base_weight': 0.2,
            'transitions': {
                'sleep': {
                    'trigger': 'stamina',
                    'probability': lambda needs: max(0, (80 - needs['stamina']) / 80.0),
                    'min_threshold': 20
                },
                'idle': {
                    'trigger': 'stamina',
                    'probability': lambda needs: needs['stamina'] / 100.0,
                    'min_threshold': 60
                }
            }
        },
        'sleep': {
            'base_weight': 0.1,
            'force_threshold': {
                'trigger': 'stamina',
                'value': 10
            },
            'transitions': {
                'idle': {
                    'trigger': 'stamina',
                    'probability': lambda needs: needs['stamina'] / 100.0,
                    'min_threshold': 80
                }
            }
        }
    }
    
    def __init__(self, pet_context, needs_manager):
        """
        Initializes the BehaviorManager.

        Args:
            pet_context (PetContext): methods to retrieve pet's current internal state
            needs_manager (NeedsManager): The NeedsManager instance (used by behaviors to manage rates)
        """
        self.pet_context = pet_context
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
        if self.current_behavior:
            old_behavior = self.current_behavior
            self.current_behavior.stop()
        self.current_behavior = self.behaviors[new_behavior_name]
        self.current_behavior.start()

        global_event_dispatcher.dispatch_event_sync(Event("behavior:changed", {
            "old_behavior": old_behavior.__class__.__name__ if old_behavior else None,
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

        current_needs = self.pet_context.get_current_needs()
        current_mood = self.pet_context.get_current_mood()
        recent_emotions = self.pet_context.get_recent_emotions()
        
        new_behavior = self._calculate_behavior(event_type, event_data, current_needs, current_mood, recent_emotions)

        if new_behavior != self.current_behavior.__class__.__name__.lower():
            self.change_behavior(new_behavior)

    def _calculate_behavior(self, event_type, event_data, current_needs, current_mood, recent_emotions):
        """
        Calculates the appropriate behavior based on the current state and event using a probabilistic transition system.

        Args:
            event_type (str): The type of event that triggered the behavior calculation.
            event_data (dict): Additional data associated with the event.
            current_needs (dict): The current state of the pet's needs.
            current_mood (Mood): The current mood of the pet.
            recent_emotions (list): A list of recent EmotionalVector objects.

        Returns:
            str: The name of the selected behavior.
        """
        if self.is_locked():
            return self.current_behavior.__class__.__name__.lower()
        
        current = self.current_behavior.__class__.__name__.lower()
        pattern = self.BEHAVIOR_PATTERNS[current]
        
        for behavior, data in self.BEHAVIOR_PATTERNS.items():
            if 'force_threshold' in data:
                threshold = data['force_threshold']
                if current_needs[threshold['trigger']] <= threshold['value']:
                    return behavior

        transition_weights = {}
        for next_behavior, rules in pattern['transitions'].items():
            if current_needs[rules['trigger']] >= rules['min_threshold']:
                prob = rules['probability'](current_needs)
                if 'condition' in rules and not rules['condition'](current_needs):
                    continue
                transition_weights[next_behavior] = prob * random.random()
        
        transition_weights[current] = pattern['base_weight'] * random.random()
        
        return max(transition_weights.items(), key=lambda x: x[1])[0]

    def get_current_behavior(self):
        """
        Returns the current behavior.

        Returns:
            Behavior: The current behavior instance.
        """
        return self.current_behavior