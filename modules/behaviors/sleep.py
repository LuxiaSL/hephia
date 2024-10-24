# modules/behaviors/sleep.py

from .behavior import Behavior
from config import Config
from event_dispatcher import global_event_dispatcher, Event

class SleepBehavior(Behavior):
    """
    Represents the pet's sleeping state, triggered when stamina hits 0 or by user clicking rest
    """

    def __init__(self, behavior_manager):
        super().__init__(behavior_manager)

    def start(self):
        super().start()
        print("SleepBehavior started.")
        self.apply_need_modifiers()

    def update(self):
        super().update()
        if not self.active:
            return
        
    def stop(self):
        print("SleepBehavior stopped.")
        self.remove_need_modifiers()
        super().stop()

    def apply_need_modifiers(self):
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in Config.SLEEP_NEED_MODIFIERS.items():
            needs_manager.alter_base_rate(need, modifier)
        
        global_event_dispatcher.dispatch_event_sync(Event("behavior:sleep:modifiers_applied", Config.SLEEP_NEED_MODIFIERS))

    def remove_need_modifiers(self):
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in Config.SLEEP_NEED_MODIFIERS.items():
            needs_manager.alter_base_rate(need, -modifier)
        
        global_event_dispatcher.dispatch_event_sync(Event("behavior:sleep:modifiers_removed", Config.SLEEP_NEED_MODIFIERS))