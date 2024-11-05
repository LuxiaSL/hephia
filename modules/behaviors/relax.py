# modules/behaviors/relax.py

from .behavior import Behavior
from config import Config
from event_dispatcher import global_event_dispatcher, Event

class RelaxBehavior(Behavior):
    """
    Represents the pet's relaxing state, catching breath after activity or plain zoning out
    """

    def __init__(self, behavior_manager):
        super().__init__(behavior_manager)
        self.name = "relax"

    def start(self):
        super().start()
        print("RelaxBehavior started.")
        self.apply_need_modifiers()

    def update(self):
        super().update()
        if not self.active:
            return
        
    def stop(self):
        print("RelaxBehavior stopped.")
        self.remove_need_modifiers()
        super().stop()

    def apply_need_modifiers(self):
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in Config.RELAX_NEED_MODIFIERS.items():
            needs_manager.alter_base_rate(need, modifier)
        
        global_event_dispatcher.dispatch_event_sync(Event("behavior:relax:modifiers_applied", Config.RELAX_NEED_MODIFIERS))

    def remove_need_modifiers(self):
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in Config.RELAX_NEED_MODIFIERS.items():
            needs_manager.alter_base_rate(need, -modifier)
        
        global_event_dispatcher.dispatch_event_sync(Event("behavior:relax:modifiers_removed", Config.RELAX_NEED_MODIFIERS))