# visualization/pet_widget.py

from PyQt5.QtWidgets import QWidget, QMenu, QAction, QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QMouseEvent
from pet.pet import Pet
from pet.movement import IdleMovement, MoveMovement
from utils.vector import Vector2D
from visualization.renderer import Renderer
import random

class PetWidget(QWidget):
    """
    PetWidget serves as the main interface between the pet's logic and its visual representation.
    """

    def __init__(self):
        super().__init__()
        self.pet = Pet()
        self.renderer = Renderer(self.pet.state, self)
        self.dragging = False
        self.drag_position = None

        self.init_ui()
        self.setup_timers()

    def init_ui(self):
        """Initializes the UI components of the widget."""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(250, 250)
        self.show()

    def setup_timers(self):
        """Sets up timers for updating the pet and its needs."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_pet)
        self.update_timer.start(50)  # Update every 50ms

        self.needs_timer = QTimer()
        self.needs_timer.timeout.connect(self.update_needs)
        self.needs_timer.start(5000)  # Update needs every 5 seconds

    def paintEvent(self, event):
        """Handles the paint event by delegating to the renderer."""
        self.renderer.render(event)

    def update_pet(self):
        """Updates the pet's state and position."""
        if not self.dragging:
            if not self.pet.state.current_movement:
                self.pet.state.current_movement = IdleMovement(self.pet.state)
                self.pet.state.current_movement.start()
            self.pet.state.current_movement.update()
            self.decide_next_action()

        # Update the widget's position based on pet_state.position
        self.move(int(self.pet.state.position.x), int(self.pet.state.position.y))
        self.update()

    def decide_next_action(self):
        """Decides the pet's next action based on its state."""
        if isinstance(self.pet.state.current_movement, IdleMovement):
            boredom_factor = self.pet.state.boredom / 100
            move_chance = 0.01 + boredom_factor * 0.1
            if random.random() < move_chance:
                self.pet.state.current_movement.stop()
                self.pet.state.current_movement = MoveMovement(self.pet.state)
                self.pet.state.current_movement.start()
        elif isinstance(self.pet.state.current_movement, MoveMovement):
            if random.random() < 0.02:
                self.pet.state.current_movement.stop()
                self.pet.state.current_movement = IdleMovement(self.pet.state)
                self.pet.state.current_movement.start()

    def update_needs(self):
        """Updates the pet's needs based on its activity."""
        if isinstance(self.pet.state.current_movement, IdleMovement):
            activity_level = 0.5
        else:
            activity_level = 1.0
        self.pet.update_needs(activity_level)

    def mousePressEvent(self, event):
        """Handles mouse press events for dragging and context menu."""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.globalPos())

    def mouseMoveEvent(self, event):
        """Handles mouse move events for dragging."""
        if self.dragging:
            self.move(event.globalPos() - self.drag_position)
            self.pet.state.position = Vector2D(self.x(), self.y())
            event.accept()

    def mouseReleaseEvent(self, event):
        """Handles mouse release events to stop dragging."""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            event.accept()

    def show_context_menu(self, position):
        """Displays the context menu at the given position."""
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
        """Displays the pet's current stats."""
        stats = f"""
        Hunger: {int(self.pet.state.hunger)}%
        Thirst: {int(self.pet.state.thirst)}%
        Boredom: {int(self.pet.state.boredom)}%
        Stamina: {int(self.pet.state.stamina)}%
        """
        print(stats)  # Placeholder for actual UI display

    def feed_pet(self):
        """Feeds the pet."""
        self.pet.feed()
        self.update()

    def give_water(self):
        """Gives water to the pet."""
        self.pet.give_water()
        self.update()

    def play_with_pet(self):
        """Plays with the pet."""
        self.pet.play()
        self.update()

    def pet_rest(self):
        """Allows the pet to rest."""
        self.pet.rest()
        self.update()
