from PyQt5.QtWidgets import QWidget, QApplication, QMenu, QAction
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QFont
from abc import ABC, abstractmethod
from enum import Enum
from pet import Pet
import random
import math

class Vector2D:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)

class PetAction(Enum):
    IDLE = 1
    MOVE = 2
    FALL = 3

class Movement:
    def __init__(self, pet_widget):
        self.pet_widget = pet_widget
        self.center_position = Vector2D(pet_widget.x() + pet_widget.width // 2, 
                                        pet_widget.y() + pet_widget.height // 2)

    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def paint(self, painter, head, body):
        pass

class IdleMovement(Movement):
    def __init__(self, pet_widget):
        super().__init__(pet_widget)
        self.idle_bob_offsets = [0] * 4
        self.idle_bob_speeds = [random.uniform(0.1, 0.3) for _ in range(4)]
        self.idle_bob_amplitudes = [random.uniform(3, 7) for _ in range(4)]
        self.idle_bob_phases = [random.uniform(0, 2 * math.pi) for _ in range(4)]

    def update(self):
        for i in range(len(self.idle_bob_offsets)):
            self.idle_bob_phases[i] += self.idle_bob_speeds[i]
            self.idle_bob_offsets[i] = self.idle_bob_amplitudes[i] * math.sin(self.idle_bob_phases[i])
        self.pet_widget.update()

    def paint(self, painter, head, body):
        for i, offset in enumerate(self.idle_bob_offsets):
            x_pos = int(self.pet_widget.width // 2 - (len(self.idle_bob_offsets) - i - 1) * 20)
            y_pos = int(self.pet_widget.height // 2 + offset)
            segment = head if i == 3 else body
            painter.drawText(x_pos, y_pos, segment)

class MoveMovement(Movement):
    def __init__(self, pet_widget):
        super().__init__(pet_widget)
        self.velocity = Vector2D(0, 0)
        self.center_position = Vector2D(pet_widget.x() + pet_widget.width // 2, 
                                        pet_widget.y() + pet_widget.height // 2)
        self.movement_duration = 0
        self.elapsed_time = 0

    def start(self):
        speed = random.uniform(1, 3)
        if random.random() < 0.8:  # 80% chance of horizontal movement
            angle = random.choice([0, 180])
        else:
            angle = random.uniform(0, 360)
        self.velocity = Vector2D(speed * math.cos(math.radians(angle)),
                                 speed * math.sin(math.radians(angle)))
        self.movement_duration = random.randint(50, 300)  # 2.5-15 seconds (assuming 50ms updates)
        self.elapsed_time = 0

    def update(self):
        self.center_position += self.velocity
        self.handle_boundaries()
        self.pet_widget.move(int(self.center_position.x - self.pet_widget.width // 2),
                             int(self.center_position.y - self.pet_widget.height // 2))
        
        self.elapsed_time += 1
        if self.elapsed_time >= self.movement_duration:
            self.pet_widget.set_action(PetAction.IDLE)

    def handle_boundaries(self):
        screen_rect = QApplication.desktop().screenGeometry()
        half_width = self.pet_widget.width // 2
        half_height = self.pet_widget.height // 2

        if self.center_position.x - half_width <= 0 or self.center_position.x + half_width >= screen_rect.width():
            self.velocity.x *= -1
            if self.center_position.x - half_width <= 0:
                self.center_position.x = half_width
            else:
                self.center_position.x = screen_rect.width() - half_width

        if self.center_position.y - half_height <= 0 or self.center_position.y + half_height >= screen_rect.height():
            self.velocity.y *= -1
            if self.center_position.y - half_height <= 0:
                self.center_position.y = half_height
            else:
                self.center_position.y = screen_rect.height() - half_height

    def paint(self, painter, head, body):
        direction = 1 if self.velocity.x >= 0 else -1
        segments =  [body] * 3 + [head]
        for i, segment in enumerate(segments):
            x_pos = int(self.pet_widget.width // 2 + direction * i * 20)
            y_pos = int(self.pet_widget.height // 2)
            painter.drawText(x_pos, y_pos, segment)


class PetWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.pet = Pet()
        self.height = 250
        self.width = 250
        self.center_position = Vector2D(self.width // 2, self.height // 2)
        self.init_ui()
        self.drag_position = None
        self.dragging = False  # Track dragging state

        self.action = PetAction.IDLE
        self.movement = IdleMovement(self)
        self.movement.start()  # Start with idle movement

        # Timers for update and needs
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_pet)
        self.update_timer.start(50)  # Update every 50ms

        self.needs_timer = QTimer()
        self.needs_timer.timeout.connect(self.update_needs)
        self.needs_timer.start(5000)  # Update needs every 5 seconds

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(self.width, self.height)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        font = QFont('Arial', 32)
        painter.setFont(font)

        head = self.pet.emotional_state
        body = '🟢'

        self.movement.paint(painter, head, body)

    def update_pet(self):
        # Update the pet's state and decide on next action.
        if not self.dragging:
            self.movement.update()
            self.decide_next_action()

    def decide_next_action(self):
        if self.action == PetAction.IDLE:
            boredom_factor = self.pet.boredom / 100
            move_chance = 0.01 + boredom_factor * 0.1
            if random.random() < move_chance:
                self.set_action(PetAction.MOVE)
        elif self.action == PetAction.MOVE:
            if random.random() < 0.02:  # 2% chance to stop moving
                self.set_action(PetAction.IDLE)
            elif random.random() < 0.05:  # 5% chance to change direction or speed
                self.movement.start()  # Reinitialize movement parameters

    def set_action(self, action):
        self.action = action
        if action == PetAction.IDLE:
            self.movement = IdleMovement(self)
        elif action == PetAction.MOVE:
            self.movement = MoveMovement(self)
        # Add more movement types as needed
        self.movement.start()

    def update_needs(self):
        # Adjust needs decay rate based on action
        if self.action == PetAction.IDLE:
            activity_level = 0.5  # Needs decay slower when idle
        else:
            activity_level = 1.0  # Needs decay faster when moving

        self.pet.update_needs(activity_level)

    def mousePressEvent(self, event):
        # Handle mouse press events for dragging and context menu.
        if event.button() == Qt.LeftButton:
            self.dragging = True  # Begin dragging
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.globalPos())

    def mouseReleaseEvent(self, event):
        # Handle mouse release events to stop dragging.
        if event.button() == Qt.LeftButton:
            self.dragging = False  # End dragging
            event.accept()

    def show_context_menu(self, position):
        menu = QMenu()

        stats_action = QAction('Stats', self)
        stats_action.triggered.connect(self.show_stats)
        menu.addAction(stats_action)

        feed_action = QAction('Feed', self)
        feed_action.triggered.connect(self.feed_pet)
        menu.addAction(feed_action)

        water_action = QAction('Give Water', self)
        water_action.triggered.connect(self.give_water)
        menu.addAction(water_action)

        play_action = QAction('Play', self)
        play_action.triggered.connect(self.play_with_pet)
        menu.addAction(play_action)

        rest_action = QAction('Rest', self)
        rest_action.triggered.connect(self.pet_rest)
        menu.addAction(rest_action)

        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(QApplication.instance().quit)
        menu.addAction(quit_action)

        menu.exec_(position)

    def show_stats(self):
        # Display current stats in a simple message box
        stats = f"""
        Hunger: {int(self.pet.hunger)}%
        Thirst: {int(self.pet.thirst)}%
        Boredom: {int(self.pet.boredom)}%
        Stamina: {int(self.pet.stamina)}%
        """
        print(stats)  # Placeholder for actual UI display

    def feed_pet(self):
        self.pet.feed()
        self.update()

    def give_water(self):
        self.pet.give_water()
        self.update()

    def play_with_pet(self):
        self.pet.play()
        self.update()

    def pet_rest(self):
        self.pet.rest()
        self.update()
