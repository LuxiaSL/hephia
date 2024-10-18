# modules/emotions/emotion.py

class Emotion:
    """
    Represents an individual emotional response using a valence-arousal model.
    """
    def __init__(self, name, valence, arousal, intensity):
        """
        Initializes an Emotion instance.

        Args:
            name (str): The name of the emotion (e.g., 'joy', 'sadness').
            valence (float): The valence of the emotion (-1 to 1).
            arousal (float): The arousal level of the emotion (-1 to 1).
            intensity (float): The intensity of the emotion (0 to 1).
        """
        self.name = name
        self.valence = valence
        self.arousal = arousal
        self.intensity = intensity

    def __repr__(self):
        return f"Emotion(name={self.name}, valence={self.valence}, arousal={self.arousal}, intensity={self.intensity})"