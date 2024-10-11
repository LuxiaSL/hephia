# modules/needs/needs_manager.py

from .need import Need
from config import Config

class NeedsManager:
    """
    Manages all of the pet's needs.
    """

    def __init__(self):
        """
        Initializes the NeedsManager with all defined needs.
        """
        self.needs = {}
        self.initialize_needs()

    def initialize_needs(self):
        """
        Initializes the needs based on the configuration.
        """
        self.needs['hunger'] = Need(
            name='hunger',
            value=Config.INITIAL_HUNGER,
            base_decay_rate=Config.HUNGER_BASE_DECAY_RATE
        )
        self.needs['thirst'] = Need(
            name='thirst',
            value=Config.INITIAL_THIRST,
            base_decay_rate=Config.THIRST_BASE_DECAY_RATE
        )
        self.needs['boredom'] = Need(
            name='boredom',
            value=Config.INITIAL_BOREDOM,
            base_decay_rate=Config.BOREDOM_BASE_DECAY_RATE
        )
        self.needs['stamina'] = Need(
            name='stamina',
            value=Config.INITIAL_STAMINA,
            base_decay_rate=Config.STAMINA_BASE_DECAY_RATE,
            min_value=0.0,
            max_value=100.0
        )

    def update_needs(self, needs_to_update=None):
        """
        Updates specified needs.

        Args:
            needs_to_update (list, optional): List of need names to update. If None, updates all needs.
        """
        if needs_to_update is None:
            needs_to_update = self.needs.keys()

        for need_name in needs_to_update:
            need = self.needs.get(need_name)
            if need:
                need.update()

    def alter_need(self, need_name, amount):
        """
        Alters a specific need by a given amount.

        Args:
            need_name (str): The name of the need to alter.
            amount (float): The amount to change the need's value by.
        """
        need = self.needs.get(need_name)
        if need:
            need.alter(amount)
        else:
            raise ValueError(f"Need '{need_name}' does not exist.")

    def get_need_value(self, need_name):
        """
        Retrieves the current value of a specific need.

        Args:
            need_name (str): The name of the need.

        Returns:
            float: The current value of the need.
        """
        need = self.needs.get(need_name)
        if need:
            return need.value
        else:
            raise ValueError(f"Need '{need_name}' does not exist.")

    def alter_base_decay_rate(self, need_name, amount):
        """
        Alters the base decay rate of a specific need.

        Args:
            need_name (str): The name of the need.
            amount (float): The amount to change the base decay rate by.
        """
        need = self.needs.get(need_name)
        if need:
            need.alter_base_decay_rate(amount)
        else:
            raise ValueError(f"Need '{need_name}' does not exist.")

    def alter_decay_rate_multiplier(self, need_name, factor):
        """
        Alters the decay rate multiplier of a specific need.

        Args:
            need_name (str): The name of the need.
            factor (float): The factor to multiply the current multiplier by.
        """
        need = self.needs.get(need_name)
        if need:
            need.alter_decay_rate_multiplier(factor)
        else:
            raise ValueError(f"Need '{need_name}' does not exist.")

    # Observer management methods remain the same...
    def subscribe_to_need(self, need_name, observer):
        need = self.needs.get(need_name)
        if need:
            need.subscribe(observer)
        else:
            raise ValueError(f"Need '{need_name}' does not exist.")

    def unsubscribe_from_need(self, need_name, observer):
        need = self.needs.get(need_name)
        if need:
            need.unsubscribe(observer)
        else:
            raise ValueError(f"Need '{need_name}' does not exist.")
