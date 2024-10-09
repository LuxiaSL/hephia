from PyQt5.QtWidgets import QWidget, QApplication, QMenu, QAction
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QFont
from pet import Pet
import random

class PetWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.pet = Pet()
        self.init_ui()
        self.moving_right = True
        self.drag_position = None

        # Timers for movement and needs update
        self.move_timer = QTimer()
        self.move_timer.timeout.connect(self.move_pet)
        self.move_timer.start(100)

        self.needs_timer = QTimer()
        self.needs_timer.timeout.connect(self.update_needs)
        self.needs_timer.start(5000)  # Update needs every 5 seconds

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(100, 100)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        font = QFont('Arial', 32)
        painter.setFont(font)

        # Draw the pet avatar
        head = self.pet.emotional_state
        body = '🟢' * 3
        avatar = head + body

        painter.drawText(self.rect(), Qt.AlignCenter, avatar)

    def move_pet(self):
        # Simple movement logic
        x, y = self.x(), self.y()
        screen_width = QApplication.desktop().screenGeometry().width()
        screen_height = QApplication.desktop().screenGeometry().height()

        if self.moving_right:
            x += 1
            if x + self.width() >= screen_width:
                self.moving_right = False
        else:
            x -= 1
            if x <= 0:
                self.moving_right = True

        # Vertical movement (occasionally)
        if random.randint(0, 100) == 0:
            y += random.choice([-50, 50])
            y = max(0, min(y, screen_height - self.height()))

        self.move(x, y)

    def update_needs(self):
        self.pet.update_needs()
        self.update()  # Refresh the avatar

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.globalPos())

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
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
