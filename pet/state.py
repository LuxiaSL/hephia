# pet/state.py

from utils.vector import Vector2D

class PetState:
    """
    PetState centralizes all attributes of the pet,
    acting as a single source of truth for the pet's state.
    """

    def __init__(self):
        """
        Initializes the PetState with default values.
        """
        # Needs
        self.hunger = 50.0
        self.thirst = 50.0
        self.boredom = 50.0
        self.stamina = 100.0

        # Emotional state
        self.emotional_state = '😊'  # Default emotional state

        # Position and Movement
        self.position = Vector2D(0, 0)
        self.direction = Vector2D(1, 0)  # Default direction

        # Current movement behavior
        self.current_movement = None

    def update_emotional_state(self):
        """
        Updates the emotional state based on current needs.
        """
        if self.hunger > 80 or self.thirst > 80:
            self.emotional_state = '😢'
        elif self.boredom > 80:
            self.emotional_state = '🥱'
        elif self.stamina < 20:
            self.emotional_state = '😴'
        else:
            self.emotional_state = '😊'
