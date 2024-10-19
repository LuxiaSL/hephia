# pet/pet_state.py

"""
PetState module.

Manages the pet's external state related to visualization, such as position and movement
within the environment. It does not influence the pet's internal state or cognitive processes.
"""

class PetState:
    """
    Encapsulates the pet's external state within the environment.

    Used exclusively by visualization components to render the pet on the screen.
    It includes attributes like position and movement direction.
    """

    def __init__(self):
        """
        Initializes the PetState with default values for external state.
        """
        # Position attributes
        self.position = (0, 0)  # (x, y) coordinates
        self.direction = (1, 0)  # (dx, dy) direction vector

        # Additional external state attributes can be added here
        # e.g., animation states, visual effects, etc.

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

    # No methods or attributes related to internal state should be added here.
