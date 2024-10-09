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
        self.state.hunger += random.uniform(0.1, 0.5) * activity_level
        self.state.thirst += random.uniform(0.1, 0.5) * activity_level
        self.state.boredom += random.uniform(0.1, 0.5) * activity_level
        self.state.stamina -= random.uniform(0.1, 0.5) * activity_level

        # Keep values within 0-100%
        self.state.hunger = clamp(self.state.hunger, 0, 100)
        self.state.thirst = clamp(self.state.thirst, 0, 100)
        self.state.boredom = clamp(self.state.boredom, 0, 100)
        self.state.stamina = clamp(self.state.stamina, 0, 100)

        self.state.update_emotional_state()

    def feed(self):
        """
        Feeds the pet, reducing hunger.
        """
        self.state.hunger -= 20
        self.state.hunger = clamp(self.state.hunger, 0, 100)
        self.state.update_emotional_state()

    def give_water(self):
        """
        Gives water to the pet, reducing thirst.
        """
        self.state.thirst -= 20
        self.state.thirst = clamp(self.state.thirst, 0, 100)
        self.state.update_emotional_state()

    def play(self):
        """
        Plays with the pet, reducing boredom and stamina.
        """
        self.state.boredom -= 20
        self.state.boredom = clamp(self.state.boredom, 0, 100)
        self.state.stamina -= 10
        self.state.stamina = clamp(self.state.stamina, 0, 100)
        self.state.update_emotional_state()

    def rest(self):
        """
        Allows the pet to rest, increasing stamina.
        """
        self.state.stamina += 20
        self.state.stamina = clamp(self.state.stamina, 0, 100)
        self.state.update_emotional_state()
