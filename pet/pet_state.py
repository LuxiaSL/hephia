# pet/pet_state.py

class PetState:
    """
    Encapsulates the pet's state information.
    """

    def __init__(self):
        """
        Initializes the PetState with default values.
        """
        # Position attributes
        self.position = (0, 0)  # (x, y) coordinates
        self.direction = (1, 0)  # (dx, dy) direction vector

        # Emotional state
        self.emotional_state = 'neutral'  # Placeholder for emotional states like 'happy', 'sad'

        # Additional state attributes can be added here
        # e.g., health status, energy level, etc.

    def update_position(self, new_position):
        """
        Updates the pet's position.

        Args:
            new_position (tuple): The new (x, y) position.
        """
        self.position = new_position

    def update_direction(self, new_direction):
        """
        Updates the pet's movement direction.

        Args:
            new_direction (tuple): The new (dx, dy) direction vector.
        """
        self.direction = new_direction

    def update_emotional_state(self, new_state):
        """
        Updates the pet's emotional state.

        Args:
            new_state (str): The new emotional state.
        """
        self.emotional_state = new_state
