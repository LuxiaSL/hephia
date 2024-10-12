# visualization/dialogs/stats_dialog.py

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton
from PyQt5.QtCore import Qt
from event_dispatcher import global_event_dispatcher, Event

class StatsDialog(QDialog):
    """
    StatsDialog displays the pet's current needs as progress bars with percentage indicators.
    """
    
    def __init__(self, needs_manager, parent=None):
        """
        Initializes the StatsDialog.

        Args:
            needs_manager (NeedsManager): Reference to the NeedsManager to retrieve current need values.
            parent (QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self.needs_manager = needs_manager
        self.setWindowTitle("Pet Stats")
        self.setFixedSize(300, 300)  # Increased height to accommodate emotional state
        self.init_ui()
        self.setup_event_listeners()
        self.initialize_emotional_state()

    def init_ui(self):
        """Initializes the UI components of the dialog."""
        layout = QVBoxLayout()

        # Create progress bars for each need
        self.hunger_bar = self.create_progress_bar("Hunger")
        self.thirst_bar = self.create_progress_bar("Thirst")
        self.boredom_bar = self.create_progress_bar("Boredom")
        self.stamina_bar = self.create_progress_bar("Stamina")

        layout.addWidget(self.hunger_bar['label'])
        layout.addWidget(self.hunger_bar['bar'])
        layout.addWidget(self.thirst_bar['label'])
        layout.addWidget(self.thirst_bar['bar'])
        layout.addWidget(self.boredom_bar['label'])
        layout.addWidget(self.boredom_bar['bar'])
        layout.addWidget(self.stamina_bar['label'])
        layout.addWidget(self.stamina_bar['bar'])

        # Display emotional state
        self.emotional_state_label = QLabel("Emotional State: ")
        self.emotional_state_label.setAlignment(Qt.AlignCenter)
        font = self.emotional_state_label.font()
        font.setPointSize(12)
        font.setBold(True)
        self.emotional_state_label.setFont(font)
        layout.addWidget(self.emotional_state_label)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def create_progress_bar(self, name):
        """
        Creates a labeled progress bar for a specific need.

        Args:
            name (str): The name of the need.

        Returns:
            dict: Contains the QLabel and QProgressBar objects.
        """
        label = QLabel(f"{name}:")
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        
        # Retrieve the current need value and normalize it to 0-100
        try:
            need_value = self.needs_manager.get_need_value(name.lower())
            initial_value = max(0, min(int(need_value), 100))
        except Exception as e:
            print(f"Error retrieving value for {name}: {e}")
            initial_value = 50  # Default value
        
        progress_bar.setValue(initial_value)
        progress_bar.setFormat("%p%")
        progress_bar.setTextVisible(True)
        progress_bar.setAlignment(Qt.AlignCenter)
        progress_bar.setStyleSheet(self.get_style_sheet(initial_value))
        return {'label': label, 'bar': progress_bar}

    def setup_event_listeners(self):
        """
        Sets up event listeners for need changes and emotional state changes.
        """
        global_event_dispatcher.add_listener("need:changed", self.on_need_changed)
        global_event_dispatcher.add_listener("pet:emotional_state_changed", self.on_emotional_state_changed)

    def initialize_emotional_state(self):
        """
        Initializes the emotional state label based on current needs.
        """
        # Request the current emotional state
        global_event_dispatcher.dispatch_event_sync(Event("ui:request_emotional_state"))

    def on_need_changed(self, event):
        """
        Updates the corresponding progress bar when a need changes.

        Args:
            event (Event): The need change event.
        """
        need_name = event.data['need_name']
        new_value = event.data['new_value']
        
        try:
            value = max(0, min(int(new_value), 100))  # Clamp between 0 and 100
        except Exception as e:
            print(f"Error normalizing value for {need_name}: {e}")
            value = 50  # Default value in case of error

        if need_name == 'hunger':
            self.hunger_bar['bar'].setValue(value)
            self.hunger_bar['bar'].setStyleSheet(self.get_style_sheet(value))
        elif need_name == 'thirst':
            self.thirst_bar['bar'].setValue(value)
            self.thirst_bar['bar'].setStyleSheet(self.get_style_sheet(value))
        elif need_name == 'boredom':
            self.boredom_bar['bar'].setValue(value)
            self.boredom_bar['bar'].setStyleSheet(self.get_style_sheet(value))
        elif need_name == 'stamina':
            self.stamina_bar['bar'].setValue(value)
            self.stamina_bar['bar'].setStyleSheet(self.get_style_sheet(value))

    def on_emotional_state_changed(self, event):
        """
        Updates the emotional state label based on the pet's current state.

        Args:
            event (Event): The emotional state change event.
        """
        new_state = event.data['new_state']
        self.emotional_state_label.setText(f"Emotional State: {new_state.capitalize()}")

    def get_style_sheet(self, value):
        """
        Returns a style sheet string based on the need value.

        Args:
            value (int): The current value of the need.

        Returns:
            str: The style sheet for the progress bar.
        """
        if value < 30:
            color = "red"
        elif value < 70:
            color = "yellow"
        else:
            color = "green"
        return f"""
            QProgressBar {{
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                width: 20px;
            }}
        """