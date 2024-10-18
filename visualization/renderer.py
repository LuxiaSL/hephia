from PyQt5.QtCore import QObject, pyqtSignal

class Renderer(QObject):
    update_pet_signal = pyqtSignal()

    def __init__(self, pet_state):
        super().__init__()
        self.pet_state = pet_state

    def update(self):
        # This method can be called to trigger a redraw of the pet
        self.update_pet_signal.emit()

    def handle_event(self, event_type, event_data=None):
        if event_type == 'mood_change':
            self.pet_state.update_mood(event_data['new_state'])
            self.update()
        elif event_type == 'action_perform':
            self.update()
        # Add more event handling as needed

    def get_emoji(self):
        # Return the appropriate emoji based on the pet's emotional state
        emojis = {
            'happy': '😊',
            'hungry': '😢',
            'bored': '🥱',
            'tired': '😴',
            'neutral': '😐'
        }
        return emojis.get(self.pet_state.mood, '😐')

    def get_body_color(self):
        # Return the appropriate body color based on the pet's state
        # This is just an example, you can implement your own logic
        return (0, 255, 0)  # Green

    def get_head_color(self):
        # Return the appropriate head color based on the pet's state
        return (255, 255, 255)  # White
