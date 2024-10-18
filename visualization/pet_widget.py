# visualization/pet_widget.py

from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QMenu, QAction, QApplication, QDesktopWidget, QFrame
from PyQt5.QtCore import Qt, QTimer, QRect, QRectF
from visualization.pet_graphics_item import PetGraphicsItem
from visualization.dialogs.stats_dialog import StatsDialog
from pet.pet import Pet
from event_dispatcher import global_event_dispatcher, Event
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
        self.setup_event_listeners()

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

    def setup_timers(self):
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_pet)
        self.update_timer.start(50)

        self.pet_timer = QTimer()
        self.pet_timer.timeout.connect(self.pet.update)
        self.pet_timer.start(1000)

    def setup_event_listeners(self):
        global_event_dispatcher.add_listener("action:completed", self.on_action_completed)
        global_event_dispatcher.add_listener("behavior:changed", self.on_behavior_changed)
        global_event_dispatcher.add_listener("need:changed", self.on_need_changed)
        global_event_dispatcher.add_listener("pet:emotional_state_changed", self.on_emotional_state_changed)
        global_event_dispatcher.add_listener("emotion:new", self.on_new_emotion)


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

    def on_action_completed(self, event):
        action_name = event.data["action_name"]
        #print(f"PetWidget: Action '{action_name}' completed.")
        self.pet_item.handle_event('action_completed', event.data)

    def on_behavior_changed(self, event):
        new_behavior = event.data["new_behavior"]
        #print(f"PetWidget: Behavior changed to {new_behavior}")
        self.pet_item.handle_event('behavior_changed', event.data)
    
    def on_new_emotion(self, event):
        emotion = event.data["emotion"]

        ##NOTE:: IN THE FUTURE, WE WANT THIS TO ONLY TRIGGER PAST A CERTAIN INTENSITY LEVEL VISUALLY. WE WANT IT TO FLOW THROUGH THE SYSTEM, BUT THINK OF HOW IT TAKES EITHER DIRECT CHOICE OR
        ## A CERTAIN INTENSITY TO EVOKE A VISUAL EMOTIONAL RESPONSE FROM SOMEONE. MEANS WE NEED TO EITHER COGNITIVELY PUSH THE INTENSITY AND SHOW IT, OR IT IS INTENSE ENOUGH AS A STANDALONE
        #print(f"PetWidget: New emotion received: {emotion}")
        self.pet_item.handle_event('emotion_new', event.data)

    def on_need_changed(self, event):
        need_name = event.data["need_name"]
        new_value = event.data["new_value"]
        #print(f"PetWidget: Need '{need_name}' changed to {new_value}")
        self.pet_item.handle_event('need_changed', event.data)

    def on_emotional_state_changed(self, event):
        new_state = event.data["new_state"]
        #print(f"PetWidget: Emotional state changed to {new_state}")
        self.pet_item.handle_event('emotional_state_changed', event.data)

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
            global_event_dispatcher.dispatch_event_sync(Event("ui:error", {"message": str(e)}))

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
        global_event_dispatcher.dispatch_event_sync(Event("ui:settings_requested"))

    def close_application(self):
        self.pet.shutdown()
        global_event_dispatcher.dispatch_event_sync(Event("application:shutdown"))
        QApplication.instance().quit()

    def closeEvent(self, event):
        self.pet.shutdown()
        global_event_dispatcher.dispatch_event_sync(Event("application:shutdown"))
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pet = Pet()
    widget = PetWidget(pet)
    sys.exit(app.exec_())