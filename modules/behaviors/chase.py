# modules/behaviors/chase.py

from .behavior import Behavior
from config import Config
from event_dispatcher import global_event_dispatcher, Event

class ChaseBehavior(Behavior):
    """
    Represents the pet's chasing state, where it actively follows the user's cursor.
    """

    def __init__(self, behavior_manager):
        super().__init__(behavior_manager)
        self.name = "chase"

    def start(self):
        super().start()
        print("ChaseBehavior started.")
        self.apply_need_modifiers()

    def update(self):
        super().update()
        if not self.active:
            return
        
    def stop(self):
        print("ChaseBehavior stopped.")
        self.remove_need_modifiers()
        super().stop()

    def apply_need_modifiers(self):
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in Config.CHASE_NEED_MODIFIERS.items():
            needs_manager.alter_base_rate(need, modifier)
        
        global_event_dispatcher.dispatch_event_sync(Event("behavior:chase:modifiers_applied", Config.CHASE_NEED_MODIFIERS))

    def remove_need_modifiers(self):
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in Config.CHASE_NEED_MODIFIERS.items():
            needs_manager.alter_base_rate(need, -modifier)
        
        global_event_dispatcher.dispatch_event_sync(Event("behavior:chase:modifiers_removed", Config.CHASE_NEED_MODIFIERS))