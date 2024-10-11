# config.py

"""
Configuration settings for the Hephia project.
"""

class Config:
    """
    Centralized configuration management.
    """

    # Pet appearance settings
    PET_WIDTH = 250
    PET_HEIGHT = 250

    # Movement settings
    PET_MOVEMENT_SPEED_RANGE = (1, 3)        # Min and max speed
    PET_MOVEMENT_DURATION_RANGE = (50, 300)  # Min and max duration

    # Initial pet needs
    INITIAL_HUNGER = 50.0
    INITIAL_THIRST = 50.0
    INITIAL_BOREDOM = 50.0
    INITIAL_STAMINA = 100.0

    # Base decay rates
    HUNGER_BASE_DECAY_RATE = 0.2
    THIRST_BASE_DECAY_RATE = 0.2
    BOREDOM_BASE_DECAY_RATE = 0.1
    STAMINA_BASE_DECAY_RATE = -0.1  # Negative because stamina decreases over time
