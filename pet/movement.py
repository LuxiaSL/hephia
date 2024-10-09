# pet/movement.py

from abc import ABC, abstractmethod
import random
import math
from utils.vector import Vector2D
from PyQt5.QtWidgets import QApplication

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

    def update(self):
        """Updates the idle animation offsets."""
        for i in range(len(self.idle_bob_offsets)):
            self.idle_bob_phases[i] += self.idle_bob_speeds[i]
            self.idle_bob_offsets[i] = self.idle_bob_amplitudes[i] * math.sin(self.idle_bob_phases[i])

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
        speed = random.uniform(1, 3)
        if random.random() < 0.8:  # 80% chance of horizontal movement
            angle = random.choice([0, 180])
        else:
            angle = random.uniform(0, 360)
        self.velocity = Vector2D(speed * math.cos(math.radians(angle)),
                                 speed * math.sin(math.radians(angle)))
        self.movement_duration = random.randint(50, 300)  # Number of update cycles
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
        pet_width = 250  # Should be dynamic based on pet dimensions
        pet_height = 250
        half_width = pet_width // 2
        half_height = pet_height // 2

        if self.pet_state.position.x - half_width <= 0 or self.pet_state.position.x + half_width >= screen_rect.width():
            self.velocity.x *= -1
            if self.pet_state.position.x - half_width <= 0:
                self.pet_state.position.x = half_width
            else:
                self.pet_state.position.x = screen_rect.width() - half_width

        if self.pet_state.position.y - half_height <= 0 or self.pet_state.position.y + half_height >= screen_rect.height():
            self.velocity.y *= -1
            if self.pet_state.position.y - half_height <= 0:
                self.pet_state.position.y = half_height
            else:
                self.pet_state.position.y = screen_rect.height() - half_height

    def stop(self):
        """Transitions back to idle movement."""
        self.pet_state.current_movement = IdleMovement(self.pet_state)
        self.pet_state.current_movement.start()
