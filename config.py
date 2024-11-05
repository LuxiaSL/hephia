# config.py

"""
Configuration settings for the Hephia project.
"""

class Config:
    """
    Centralized configuration management.
    """
    
    # timers (in seconds)
    NEED_UPDATE_TIMER = 5 
    EMOTION_UPDATE_TIMER = 0.05

    # Pet appearance settings
    PET_WIDTH = 250
    PET_HEIGHT = 250

    # Initial pet needs
    INITIAL_HUNGER = 0
    INITIAL_THIRST = 0
    INITIAL_BOREDOM = 0
    INITIAL_COMPANIONSHIP = 0
    INITIAL_STAMINA = 100.0

    # Base decay rates (per minute)
    HUNGER_BASE_RATE = 0.0056  # Approximately 3 hours to fill (0-100)
    THIRST_BASE_RATE = 0.0056  # Approximately 3 hours to fill (0-100)
    BOREDOM_BASE_RATE = 0.0028  # Approximately 6 hours to fill (0-100)
    COMPANIONSHIP_BASE_RATE = 0.0028  # Approximately 6 hours to fill (0-100)
    STAMINA_BASE_RATE = -0.0056  # Approximately 3 hours to deplete (100-0)

    # behavior effects on needs
    IDLE_NEED_MODIFIERS = {}  # No modifiers during idle

    WALK_NEED_MODIFIERS = {
        'hunger': 0.001,
        'thirst': 0.001,
        'boredom': -0.001,
        'companionship': 0.001
    }

    CHASE_NEED_MODIFIERS = {
        'hunger': 0.0025,
        'thirst': 0.0025,
        'stamina': -0.01,  # Increased stamina use in chase
        'boredom': -0.005,
        'companionship': -0.003
    }

    RELAX_NEED_MODIFIERS = {
        'hunger': -0.0015,
        'thirst': -0.0015,
        'stamina': 0.001,  # Gradual stamina recovery
        'boredom': 0.001,
        'companionship': 0.001
    }

    SLEEP_NEED_MODIFIERS = {
        'hunger': -0.0015,
        'thirst': -0.0015,
        'stamina': 0.015,  # Rapid stamina recovery during sleep
        'boredom': 0.0002,
        'companionship': 0.0002
    }

