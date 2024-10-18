# modules/memory/memory_system.py

class BodyMemory:
    """
    Stores initial, unmediated emotional responses.
    """

    def __init__(self):
        self.emotional_log = []

    def log(self, emotion):
        """
        Logs an emotion to the body memory.

        Args:
            emotion (Emotion): The emotion to log.
        """
        self.emotional_log.append(emotion)

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

