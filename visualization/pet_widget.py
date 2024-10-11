from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QMenu, QAction, QApplication, QDesktopWidget, QFrame
from PyQt5.QtCore import Qt, QTimer, QRect, QRectF
from visualization.pet_graphics_item import PetGraphicsItem
from visualization.dialogs.stats_dialog import StatsDialog
from pet.pet import Pet
import sys

class PetWidget(QGraphicsView):
    def __init__(self, pet):
        super().__init__()
        self.pet = pet
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setFrameShape(QFrame.NoFrame)
        
        self.pet_item = PetGraphicsItem(self.pet)
        self.scene.addItem(self.pet_item)
        
        self.init_ui()
        self.setup_timers()
        self.setup_observers()

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        
        # Get the geometry of all screens
        desktop = QDesktopWidget()
        combined_geometry = QRect()
        for i in range(desktop.screenCount()):
            combined_geometry = combined_geometry.united(desktop.screenGeometry(i))
        
        # Set the widget and scene to cover all screens
        self.setGeometry(combined_geometry)
        self.setSceneRect(QRectF(combined_geometry))
        
        # Ensure the view doesn't render anything outside the scene rect
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        
        # Position the pet on a random screen
        random_screen = desktop.screen(desktop.primaryScreen())
        random_rect = random_screen.geometry()
        initial_x = random_rect.x() + random_rect.width() // 2
        initial_y = random_rect.y() + random_rect.height() // 2
        self.pet_item.setPos(initial_x, initial_y)
        
        self.show()
        
        self.show()

    def setup_timers(self):
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_pet)
        self.update_timer.start(50)

        self.pet_timer = QTimer()
        self.pet_timer.timeout.connect(self.pet.update)
        self.pet_timer.start(1000)

    def setup_observers(self):
        self.pet.action_manager.subscribe_to_action('feed', self.on_action_perform)
        self.pet.action_manager.subscribe_to_action('play', self.on_action_perform)
        self.pet.action_manager.subscribe_to_action('give_water', self.on_action_perform)
        self.pet.action_manager.subscribe_to_action('rest', self.on_action_perform)

        self.pet.behavior_manager.subscribe_to_behavior('start', self.on_behavior_event)
        self.pet.behavior_manager.subscribe_to_behavior('stop', self.on_behavior_event)
        self.pet.behavior_manager.subscribe_to_behavior('update', self.on_behavior_event)

        self.pet.needs_manager.subscribe_to_need('hunger', self.on_need_change)
        self.pet.needs_manager.subscribe_to_need('thirst', self.on_need_change)
        self.pet.needs_manager.subscribe_to_need('boredom', self.on_need_change)
        self.pet.needs_manager.subscribe_to_need('stamina', self.on_need_change)

    def update_pet(self):
        self.pet_item.update_position()
        
        # Check if the pet is out of the combined screen bounds
        pet_pos = self.pet_item.pos()
        if not self.sceneRect().contains(pet_pos):
            # Find a new position on a random screen
            desktop = QDesktopWidget()
            random_screen = desktop.screen(desktop.primaryScreen())
            random_rect = random_screen.geometry()
            new_x = random_rect.x() + random_rect.width() // 2
            new_y = random_rect.y() + random_rect.height() // 2
            self.pet_item.setPos(new_x, new_y)
        
        self.scene.update()

    def on_action_perform(self, action, event_type):
        if event_type == 'perform':
            action_name = action.__class__.__name__.replace('Action', '').lower()
            print(f"PetWidget: Action '{action_name}' performed.")
            self.pet_item.handle_event('action_perform', {'action_name': action_name})

    def on_behavior_event(self, behavior, event_type):
        print(f"PetWidget: Behavior '{behavior.__class__.__name__}' {event_type}.")
        self.pet_item.handle_event('behavior_event', {'behavior_name': behavior.__class__.__name__, 'event_type': event_type})

    def on_need_change(self, need):
        print(f"PetWidget: Need '{need.name}' changed to {need.value}.")
        self.pet_item.handle_event('emotional_change', {'new_state': self.pet.state.emotional_state})

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        context_menu.setStyleSheet("""
            QMenu {
                background-color: white;
                color: black;
            }
            QMenu::item:selected {
                background-color: lightgray;
            }
        """)

        feed_action = QAction("Feed", self)
        give_water_action = QAction("Give Water", self)
        play_action = QAction("Play", self)
        rest_action = QAction("Rest", self)
        view_stats_action = QAction("View Stats", self)
        settings_action = QAction("Settings", self)
        quit_action = QAction("Quit", self)

        context_menu.addAction(feed_action)
        context_menu.addAction(give_water_action)
        context_menu.addAction(play_action)
        context_menu.addAction(rest_action)
        context_menu.addSeparator()
        context_menu.addAction(view_stats_action)
        context_menu.addAction(settings_action)
        context_menu.addSeparator()
        context_menu.addAction(quit_action)

        feed_action.triggered.connect(lambda: self.perform_action('feed'))
        give_water_action.triggered.connect(lambda: self.perform_action('give_water'))
        play_action.triggered.connect(lambda: self.perform_action('play'))
        rest_action.triggered.connect(lambda: self.perform_action('rest'))
        view_stats_action.triggered.connect(self.show_stats_dialog)
        settings_action.triggered.connect(self.open_settings)
        quit_action.triggered.connect(self.close_application)

        context_menu.exec_(event.globalPos())

    def perform_action(self, action_name):
        try:
            self.pet.perform_action(action_name)
        except ValueError as e:
            print(f"PetWidget: {e}")

    def show_stats_dialog(self):
        stats_dialog = StatsDialog(self.pet.needs_manager, self)
        stats_dialog.setWindowFlags(stats_dialog.windowFlags() | Qt.WindowStaysOnTopHint)
        stats_dialog.setStyleSheet("""
            QDialog {
                background-color: white;
                color: black;
            }
            QLabel, QProgressBar {
                color: black;
            }
        """)
        stats_dialog.show()

    def open_settings(self):
        print("Settings dialog is not yet implemented.")

    def close_application(self):
        self.pet.shutdown()
        QApplication.instance().quit()

    def closeEvent(self, event):
        self.pet.shutdown()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pet = Pet()
    widget = PetWidget(pet)
    sys.exit(app.exec_())