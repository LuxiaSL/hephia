# visualization/pet_widget.py

from PyQt5.QtWidgets import QWidget, QMenu, QAction, QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QMouseEvent
from pet.pet import Pet
from pet.movement import IdleMovement, MoveMovement
from pet.actions import PetActions
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
        # Update pet dimensions in state
        self.pet.state.width = self.width()
        self.pet.state.height = self.height()
        self.show()

    def setup_timers(self):
        """Sets up timers for updating the pet and its needs."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_pet)
        self.update_timer.start(50)  # Update every 50ms

        self.needs_timer = QTimer()
        self.needs_timer.timeout.connect(self.update_needs)
        self.needs_timer.start(5000)  # Update needs every 5 seconds

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

        settings_action = QAction('Settings', self)
        settings_action.triggered.connect(self.open_settings)
        menu.addAction(settings_action)

        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(QApplication.instance().quit)
        menu.addAction(quit_action)

        menu.exec_(position)

    def feed_pet(self):
        """Feeds the pet."""
        PetActions.feed(self.pet, food_value=1)  # Default food_value for now

    def give_water(self):
        """Gives water to the pet."""
        PetActions.give_water(self.pet, thirst_value=1)  # Default thirst_value for now

    def play_with_pet(self):
        """Plays with the pet."""
        PetActions.play_with_pet(self.pet, play_value=1)  # Default play_value for now

    def pet_rest(self):
        """Makes the pet rest."""
        PetActions.rest_pet(self.pet)

    def open_settings(self):
        """Opens the settings dialog (placeholder)."""
        print("Settings dialog opened (placeholder)")

    def update_needs(self):
        """Updates the pet's needs."""
        self.pet.update_needs()

    def update_pet(self):
        """Updates the pet's state."""
        if self.pet.state.current_movement:
            self.pet.state.current_movement.update()
        self.renderer.update()
