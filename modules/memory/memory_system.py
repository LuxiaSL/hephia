# modules/memory/memory_system.py

class BodyMemory:
    """
    Stores initial, unmediated emotional responses.
    """

    def __init__(self, max_size=10):
        self.emotional_log = []
        self.max_size = max_size

    def log(self, emotion):
        """
        Logs an emotion to the body memory.

        Args:
            emotion (Emotion): The emotion to log.
        """
        self.emotional_log.append(emotion)
        # Keep only the last 'max_size' emotions
        if len(self.emotional_log) > self.max_size:
            self.emotional_log.pop(0)

    def get_recent_emotions(self):
        """
        Returns the list of recent emotions.

        Returns:
            list of Emotion: The recent emotions.
        """
        return self.emotional_log.copy()


class MemorySystem:
    """
    Manages different types of memory.
    """

    def __init__(self):
        """
        Initializes the MemorySystem.
        """
        self.body_memory = BodyMemory()
        # Other memory components like WorkingMemory, LongTermMemory can be added

