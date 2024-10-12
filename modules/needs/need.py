# modules/needs/need.py

class Need:
    """
    Represents a single need of the pet.
    """

    def __init__(self, name, type="physical", value=50.0, base_decay_rate=0.2, min_value=0.0, max_value=100.0):
        """
        Initializes a Need instance.

        Args:
            name (str): The name of the need (e.g., 'hunger').
            type (str): The category of need (physical, emotional, cognitive)
            value (float, optional): The initial value of the need.
            base_decay_rate (float, optional): The base decay rate per update cycle.
            min_value (float, optional): The minimum value the need can have.
            max_value (float, optional): The maximum value the need can have.
        """
        self.name = name
        self.value = value
        self.base_decay_rate = base_decay_rate
        self.min_value = min_value
        self.max_value = max_value
        self.type = type

        # Decay rate controls
        self.base_decay_rate = base_decay_rate
        self.decay_rate_multiplier = 1.0

    def update(self):
        """
        Updates the need by increasing its value based on the effective decay rate.
        """
        decay_rate = self.calculate_effective_decay_rate()
        self.alter(decay_rate)

    def alter(self, amount):
        """
        Alters the need's value by a specified amount, ensuring it stays within min and max bounds.

        Args:
            amount (float): The amount to change the need's value by.
        """
        self.value = max(self.min_value, min(self.value + amount, self.max_value))

    def calculate_effective_decay_rate(self):
        """
        Calculates the effective decay rate using the base rate and multiplier.

        Returns:
            float: The effective decay rate.
        """
        return self.base_decay_rate * self.decay_rate_multiplier

    def alter_base_decay_rate(self, amount):
        """
        Alters the base decay rate by a specified amount.

        Args:
            amount (float): The amount to change the base decay rate by.
        """
        self.base_decay_rate += amount
        self.base_decay_rate = max(0, self.base_decay_rate)

    def alter_decay_rate_multiplier(self, factor):
        """
        Alters the decay rate multiplier by a specified factor.

        Args:
            factor (float): The factor to multiply the current multiplier by.
        """
        self.decay_rate_multiplier *= factor
        self.decay_rate_multiplier = max(0, self.decay_rate_multiplier)