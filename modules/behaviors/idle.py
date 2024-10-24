# modules/behaviors/idle.py

from .behavior import Behavior
from config import Config
from event_dispatcher import global_event_dispatcher, Event

class IdleBehavior(Behavior):
    """
    Represents the pet's idle/default state.
    """

    def __init__(self, behavior_manager):
        super().__init__(behavior_manager)

    def start(self):
        super().start()
        print("IdleBehavior started.")
        self.apply_need_modifiers()

    def update(self):
        super().update()
        if not self.active:
            return
        
    def stop(self):
        print("IdleBehavior stopped.")
        self.remove_need_modifiers()
        super().stop()

    def apply_need_modifiers(self):
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in Config.IDLE_NEED_MODIFIERS.items():
            needs_manager.alter_base_rate(need, modifier)
        
        global_event_dispatcher.dispatch_event_sync(Event("behavior:idle:modifiers_applied", Config.IDLE_NEED_MODIFIERS))

    def remove_need_modifiers(self):
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in Config.IDLE_NEED_MODIFIERS.items():
            needs_manager.alter_base_rate(need, -modifier)
        
        global_event_dispatcher.dispatch_event_sync(Event("behavior:idle:modifiers_removed", Config.IDLE_NEED_MODIFIERS))