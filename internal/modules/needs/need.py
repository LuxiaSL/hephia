# modules/needs/need.py

import time
from config import Config


class Need:
    """
    Represents a single need.
    """

    def __init__(self, name, type="physical", value=50.0, base_rate=0.2, min_value=0.0, max_value=100.0):
        """
        Initializes a Need instance.

        Args:
            name (str): The name of the need (e.g., 'hunger').
            type (str): The category of need (physical, emotional, cognitive)
            value (float, optional): The initial value of the need.
            base_rate (float, optional): The base rate per update cycle.
            min_value (float, optional): The minimum value the need can have.
            max_value (float, optional): The maximum value the need can have.
        """
        self.name = name
        self.value = value
        self.base_rate = base_rate
        self.min_value = min_value
        self.max_value = max_value
        self.type = type

        self.base_rate_multiplier = 1.0
        self.urgency_curve = name != "stamina"  # stamina excluded from urgency acceleration

        # Cross-talk modifier (set externally by NeedsManager)
        self._context_modifier = 1.0

        # Afterglow state (temporary rate reduction after satisfaction)
        self._afterglow_factor = 1.0
        self._afterglow_start = 0.0
        self._afterglow_duration = 0.0

    def alter(self, amount):
        """
        Alters the need's value by a specified amount, ensuring it stays within min and max bounds.

        Args:
            amount (float): The amount to change the need's value by.
        """
        self.value = max(self.min_value, min(self.value + amount, self.max_value))

    def update(self):
        """
        Updates the need by increasing its value based on the effective rate.
        """
        rate = self.calculate_effective_rate()
        self.alter(rate)

    def calculate_effective_rate(self):
        """
        Calculates the effective rate using the base rate, multiplier,
        urgency curve, cross-talk modifier, and afterglow.

        Returns:
            float: The effective rate.
        """
        rate = self.base_rate * self.base_rate_multiplier

        # Apply urgency curve: rate accelerates as need becomes critical
        if self.urgency_curve and self.max_value > 0:
            urgency = (self.value / self.max_value) ** Config.NEED_CURVE_N
            rate *= (1.0 + Config.NEED_CURVE_K * urgency)

        # Apply cross-talk modifier (set externally by NeedsManager)
        rate *= self._context_modifier

        # Apply afterglow (temporary rate reduction after satisfaction)
        rate *= self._current_afterglow()

        return rate

    def apply_afterglow(self, factor: float, duration: float) -> None:
        """Apply temporary rate reduction after satisfaction."""
        self._afterglow_factor = factor
        self._afterglow_start = time.time()
        self._afterglow_duration = duration

    def _current_afterglow(self) -> float:
        """Get current afterglow multiplier (decays from factor back to 1.0)."""
        if self._afterglow_duration <= 0:
            return 1.0
        elapsed = time.time() - self._afterglow_start
        if elapsed >= self._afterglow_duration:
            self._afterglow_duration = 0.0
            return 1.0
        progress = elapsed / self._afterglow_duration
        return self._afterglow_factor + (1.0 - self._afterglow_factor) * progress

    def alter_base_rate(self, amount):
        """
        Alters the base rate by a specified amount.

        Args:
            amount (float): The amount to change the base rate by.
        """
        self.base_rate += amount
        self.base_rate = max(0, self.base_rate)

    def alter_rate_multiplier(self, factor):
        """
        Alters the rate multiplier by a specified factor.

        Args:
            factor (float): The factor to add to the current multiplier.
        """
        self.base_rate_multiplier += factor
        self.base_rate_multiplier = max(0, self.base_rate_multiplier)
