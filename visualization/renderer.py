# visualization/renderer.py

from PyQt5.QtGui import QPainter, QFont
from pet.movement import IdleMovement, MoveMovement

class Renderer:
    """
    Renderer handles all drawing operations for the pet,
    based on the current PetState.
    """

    def __init__(self, pet_state, widget):
        """
        Initializes the Renderer with the pet's state and the widget to draw on.

        Args:
            pet_state (PetState): The pet's current state.
            widget (QWidget): The widget to render on.
        """
        self.pet_state = pet_state
        self.widget = widget

    def render(self, event):
        """Renders the pet based on its current state."""
        painter = QPainter(self.widget)
        font = QFont('Arial', 32)
        painter.setFont(font)

        head = self.pet_state.emotional_state
        body = '🟢'

        if isinstance(self.pet_state.current_movement, IdleMovement):
            self.render_idle(painter, head, body)
        elif isinstance(self.pet_state.current_movement, MoveMovement):
            self.render_move(painter, head, body)
        else:
            self.render_idle(painter, head, body)

    def render_idle(self, painter, head, body):
        """Renders the pet in idle state."""
        idle_movement = self.pet_state.current_movement
        for i, offset in enumerate(idle_movement.idle_bob_offsets):
            x_pos = int(self.widget.width() // 2 - (len(idle_movement.idle_bob_offsets) - i - 1) * 20)
            y_pos = int(self.widget.height() // 2 + offset)
            segment = head if i == 3 else body
            painter.drawText(x_pos, y_pos, segment)

    def render_move(self, painter, head, body):
        """Renders the pet in moving state."""
        move_movement = self.pet_state.current_movement
        direction = 1 if move_movement.velocity.x >= 0 else -1
        segments = [body] * 3 + [head]
        for i, segment in enumerate(segments):
            x_pos = int(self.widget.width() // 2 + direction * i * 20)
            y_pos = int(self.widget.height() // 2)
            painter.drawText(x_pos, y_pos, segment)
