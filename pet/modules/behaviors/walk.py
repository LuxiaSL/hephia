# modules/behaviors/walk.py

from .behavior import Behavior
from config import Config
from event_dispatcher import global_event_dispatcher, Event

class WalkBehavior(Behavior):
    """
    Represents the pet's walking state.
    """

    def __init__(self, behavior_manager):
        super().__init__(behavior_manager)
        self.name = "walk"

    def start(self):
        super().start()
        print("WalkBehavior started.")
        self.apply_need_modifiers()

    def update(self):
        super().update()
        if not self.active:
            return

    def stop(self):
        print("WalkBehavior stopped.")
        self.remove_need_modifiers()
        super().stop()

    def apply_need_modifiers(self):
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in Config.WALK_NEED_MODIFIERS.items():
            needs_manager.alter_base_rate(need, modifier)
        
        global_event_dispatcher.dispatch_event_sync(Event("behavior:walk:modifiers_applied", Config.WALK_NEED_MODIFIERS))

    def remove_need_modifiers(self):
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in Config.WALK_NEED_MODIFIERS.items():
            needs_manager.alter_base_rate(need, -modifier)
        
        global_event_dispatcher.dispatch_event_sync(Event("behavior:walk:modifiers_removed", Config.WALK_NEED_MODIFIERS))