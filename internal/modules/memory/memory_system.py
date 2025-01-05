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

    def coalesce(self):
        """
        eventually, will self-review memories, send back to cognitive processor, determine whether to remember, done. something along those lines. discuss theory here.
        """
        return

    def get_memory_state(self):
        """Gets the persistent state of the memory system."""
        return {
            "body_memory": {
                "max_size": self.body_memory.max_size,
                "emotional_log": [
                    {
                        "valence": emotion.valence,
                        "arousal": emotion.arousal,
                        "intensity": emotion.intensity,
                        "name": emotion.name,
                        "source_type": emotion.source_type,
                        "timestamp": emotion.timestamp
                    }
                    for emotion in self.body_memory.emotional_log
                ]
            }
        }

    def set_memory_state(self, state):
        """Sets the memory system state."""
        if not state:
            return
            
        body_memory = state.get("body_memory", {})
        self.body_memory = BodyMemory(max_size=body_memory.get("max_size", 10))
        
        for emotion_data in body_memory.get("emotional_log", []):
            emotion = EmotionalVector(
                valence=emotion_data["valence"],
                arousal=emotion_data["arousal"],
                intensity=emotion_data["intensity"],
                name=emotion_data.get("name"),
                source_type=emotion_data.get("source_type")
            )
            if "timestamp" in emotion_data:
                emotion.timestamp = emotion_data["timestamp"]
            
            self.body_memory.log(emotion)