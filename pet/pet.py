# pet/pet.py

from .state import PetState
from utils.helpers import clamp
import random

class Pet:
    """
    The Pet class encapsulates the core logic of the pet,
    updating its state based on interactions and time.
    """

    def __init__(self, state=None):
        """
        Initializes the Pet with a given state or creates a new one.
        """
        self.state = state or PetState()

    def update_needs(self, activity_level=1.0):
        """
        Updates the pet's needs based on the activity level.

        Args:
            activity_level (float): Modifier for how quickly needs decay.
        """
        # Random decay to introduce stochastic behavior
        self.alter_need('hunger', random.uniform(0.1, 0.5) * activity_level)
        self.alter_need('thirst', random.uniform(0.1, 0.5) * activity_level)
        self.alter_need('boredom', random.uniform(0.1, 0.5) * activity_level)
        self.alter_need('stamina', -random.uniform(0.1, 0.5) * activity_level)  # Decrease stamina

        self.state.update_emotional_state()

    def alter_need(self, need, amount):
        """
        Alters a specified need by a given amount.

        Args:
            need (str): The name of the need to alter.
            amount (float): The amount to alter the need by.
        """
        if hasattr(self.state, need):
            current_value = getattr(self.state, need)
            new_value = clamp(current_value + amount, 0, 100)
            setattr(self.state, need, new_value)
        else:
            raise AttributeError(f"PetState has no attribute '{need}'")

    def feed(self, food_value=1, type=None):
        """
        Feeds the pet, reducing hunger.

        Args:
            food_value (int, optional): The strength of the food being given
            type (str, optional): The name of the item given (eventually used for favorites/dislikes)
        """
        # Placeholder logic for different food items
        hunger_reduction = -20 * food_value 
        self.alter_need('hunger', hunger_reduction)
        self.state.update_emotional_state()

    def drink(self, thirst_value=1, type=None):
        """
        Gives drink to the pet, reducing thirst.

        Args:
            thirst_value (int, optional): The strength of the drink being given
            type (str, optional): The name of the item given (eventually used for favorites/dislikes)
        """
        # Placeholder logic for different water items
        thirst_reduction = -20 * thirst_value
        self.alter_need('thirst', thirst_reduction)
        self.state.update_emotional_state()

    def play(self, play_value=1, type=None):
        """
        Plays with the pet, reducing boredom and stamina.

        Args:
            play_value (int, optional): The strength of the play being performed
            type (str, optional): The name of the play performed (eventually used for favorites/dislikes)
        """
        # Placeholder logic for different play activities
        boredom_reduction = -20 * play_value
        stamina_cost = -10 * play_value
        self.alter_need('boredom', boredom_reduction)
        self.alter_need('stamina', stamina_cost)
        self.state.update_emotional_state()

    def rest(self):
        """
        Allows the pet to rest, increasing stamina.
        """
        self.alter_need('stamina', 20)
        self.state.update_emotional_state()
