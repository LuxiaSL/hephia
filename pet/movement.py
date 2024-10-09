# pet/movement.py

from abc import ABC, abstractmethod
import random
import math
from utils.vector import Vector2D
from PyQt5.QtWidgets import QApplication
import config

class Movement(ABC):
    """
    Base class for movement behaviors.
    """

    def __init__(self, pet_state):
        """
        Initializes the Movement with a reference to the PetState.

        Args:
            pet_state (PetState): The pet's current state.
        """
        self.pet_state = pet_state

    @abstractmethod
    def start(self):
        """Starts the movement behavior."""
        pass

    @abstractmethod
    def update(self):
        """Updates the movement behavior."""
        pass

    @abstractmethod
    def stop(self):
        """Stops the movement behavior."""
        pass

class IdleMovement(Movement):
    """
    Handles idle animations and behaviors.
    """

    def __init__(self, pet_state):
        super().__init__(pet_state)
        self.idle_bob_offsets = [0] * 4
        self.idle_bob_speeds = [random.uniform(0.1, 0.3) for _ in range(4)]
        self.idle_bob_amplitudes = [random.uniform(3, 7) for _ in range(4)]
        self.idle_bob_phases = [random.uniform(0, 2 * math.pi) for _ in range(4)]

    def start(self):
        """No initialization needed for idle movement."""
        pass

    def update(self):
        """Updates the idle animation offsets."""
        for i in range(len(self.idle_bob_offsets)):
            self.idle_bob_phases[i] += self.idle_bob_speeds[i]
            self.idle_bob_offsets[i] = self.idle_bob_amplitudes[i] * math.sin(self.idle_bob_phases[i])

    def stop(self):
        """No cleanup needed for idle movement."""
        pass

class MoveMovement(Movement):
    """
    Handles moving animations and behaviors.
    """

    def __init__(self, pet_state):
        super().__init__(pet_state)
        self.velocity = Vector2D(0, 0)
        self.movement_duration = 0
        self.elapsed_time = 0

    def start(self):
        """Initializes movement parameters."""
        speed_range = config.PET_MOVEMENT_SPEED_RANGE
        speed = random.uniform(*speed_range)
        if random.random() < 0.8:  # 80% chance of horizontal movement
            angle = random.choice([0, 180])
        else:
            angle = random.uniform(0, 360)
        self.velocity = Vector2D(speed * math.cos(math.radians(angle)),
                                 speed * math.sin(math.radians(angle)))
        duration_range = config.PET_MOVEMENT_DURATION_RANGE
        self.movement_duration = random.randint(*duration_range)
        self.elapsed_time = 0

    def update(self):
        """Updates the pet's position based on velocity."""
        self.pet_state.position += self.velocity
        self.handle_boundaries()
        self.elapsed_time += 1
        if self.elapsed_time >= self.movement_duration:
            self.stop()

    def handle_boundaries(self):
        """Ensures the pet stays within screen boundaries."""
        screen_rect = QApplication.desktop().screenGeometry()
        pet_width = self.pet_state.width
        pet_height = self.pet_state.height

        if self.pet_state.position.x <= 0 or self.pet_state.position.x + pet_width >= screen_rect.width():
            self.velocity.x *= -1
            if self.pet_state.position.x <= 0:
                self.pet_state.position.x = 0
            else:
                self.pet_state.position.x = screen_rect.width() - pet_width

        if self.pet_state.position.y <= 0 or self.pet_state.position.y + pet_height >= screen_rect.height():
            self.velocity.y *= -1
            if self.pet_state.position.y <= 0:
                self.pet_state.position.y = 0
            else:
                self.pet_state.position.y = screen_rect.height() - pet_height

    def stop(self):
        """Transitions back to idle movement."""
        self.pet_state.current_movement = IdleMovement(self.pet_state)
        self.pet_state.current_movement.start()
