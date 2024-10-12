from PyQt5.QtWidgets import QGraphicsItem, QDesktopWidget
from PyQt5.QtCore import QRectF, Qt, QPointF
from PyQt5.QtGui import QPainter, QColor, QFont, QFontDatabase
from visualization.renderer import Renderer
from event_dispatcher import global_event_dispatcher, Event
import math

class PetGraphicsItem(QGraphicsItem):
    def __init__(self, pet):
        super().__init__()
        self.pet = pet
        self.renderer = Renderer(pet.state)
        self.renderer.update_pet_signal.connect(self.update)
        self.body_color = QColor(0, 255, 0)
        self.head_color = QColor(255, 255, 255)
        self.emoji_font = QFont("Segoe UI Emoji", 22)

        self.animation_phase = 0
        self.animation_speed = 0.1
        self.setZValue(1)
        self.speed = 2 

        self.setup_event_listeners()

    def setup_event_listeners(self):
        global_event_dispatcher.add_listener("pet:updated", self.on_pet_updated)
        global_event_dispatcher.add_listener("behavior:changed", self.on_behavior_changed)
        global_event_dispatcher.add_listener("pet:emotional_state_changed", self.on_emotional_state_changed)

    def boundingRect(self):
        return QRectF(-60, -60, 120, 120)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        self.body_color = QColor(*self.renderer.get_body_color())
        self.head_color = QColor(*self.renderer.get_head_color())
        self.draw_body(painter)
        self.draw_head(painter)

    def draw_body(self, painter):
        painter.setBrush(self.body_color)
        painter.setPen(Qt.NoPen)

        num_segments = 5
        segment_length = 15
        segment_width = 25

        dx, dy = self.pet.state.direction
        angle = math.atan2(dy, dx)

        for i in range(1, num_segments):  # Start from 1 to leave space for the head
            offset = i * segment_length
            x = -offset * math.cos(angle)
            y = -offset * math.sin(angle)
            
            wave_offset = math.sin(self.animation_phase + i * 0.5) * 5
            x += -wave_offset * math.sin(angle)
            y += wave_offset * math.cos(angle)
            
            painter.drawEllipse(QPointF(x, y), segment_width / 2, segment_width / 2)

        self.animation_phase += self.animation_speed


    def draw_head(self, painter):
        dx, dy = self.pet.state.direction
        head_radius = 14
        
        # Adjust head position to be slightly inward
        max_offset = 8  # Maximum offset to prevent clipping
        head_x = max(-max_offset, min(max_offset, 10 * dx))
        head_y = max(-max_offset, min(max_offset, 10 * dy))

        # Draw the head
        painter.setBrush(self.head_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(head_x, head_y), head_radius, head_radius)

        # Draw the emoji
        painter.setFont(self.emoji_font)
        emoji = self.renderer.get_emoji()
        if not emoji:
            emoji = '😊'  # Default emoji if none is set

        painter.setPen(Qt.black)
        emoji_rect = QRectF(head_x - head_radius, head_y - head_radius, 
                            head_radius * 2, head_radius * 2)
        painter.drawText(emoji_rect, Qt.AlignCenter, emoji)

    def update_position(self):
        new_x = self.x() + self.pet.state.direction[0] * self.speed
        new_y = self.y() + self.pet.state.direction[1] * self.speed

        # Get the combined geometry of all screens
        desktop = QDesktopWidget()
        combined_geometry = QRectF()
        for i in range(desktop.screenCount()):
            combined_geometry = combined_geometry.united(QRectF(desktop.screenGeometry(i)))

        # Check if the new position is within the combined screen space
        if not combined_geometry.contains(new_x, new_y):
            # Bounce off the edges
            if new_x <= combined_geometry.left() or new_x >= combined_geometry.right():
                self.pet.state.direction = (-self.pet.state.direction[0], self.pet.state.direction[1])
            if new_y <= combined_geometry.top() or new_y >= combined_geometry.bottom():
                self.pet.state.direction = (self.pet.state.direction[0], -self.pet.state.direction[1])
            
            # Recalculate new position after bouncing
            new_x = self.x() + self.pet.state.direction[0] * self.speed
            new_y = self.y() + self.pet.state.direction[1] * self.speed

        self.setPos(new_x, new_y)
        self.update()

    def on_pet_updated(self, event):
        self.update()

    def on_behavior_changed(self, event):
        new_behavior = event.data["new_behavior"]
        # Update animation or appearance based on new behavior
        self.update()

    def on_emotional_state_changed(self, event):
        new_state = event.data["new_state"]
        # Update appearance based on new emotional state
        self.update()

    def handle_event(self, event_type, event_data):
        self.renderer.handle_event(event_type, event_data)
        self.update()

    def advance(self, phase):
        if phase == 0:
            return
        self.update_position()