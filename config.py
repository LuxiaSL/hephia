# config.py

"""
Configuration settings for the Hephia project.
"""

class Config:
    """
    Centralized configuration management.
    """
    
    # internal timers (in seconds)
    NEED_UPDATE_TIMER = 5
    EMOTION_UPDATE_TIMER = 0.05

    # ExoProcessor settings
    EXO_TEMPERATURE = 0.7
    EXO_MAX_TOKENS = 150 
    EXO_LOOP_TIMER = 5.0  
    EXO_MIN_INTERVAL = 5.0  
    EXO_TIMEOUT = 30.0     
    LLM_TIMEOUT = 15.0     
    EXO_MAX_MESSAGES = 10
    
    # UI settings
    TERMINAL_WIDTH = 50
    MAX_RECENT_EMOTIONS = 3

    # Pet appearance settings
    PET_WIDTH = 250
    PET_HEIGHT = 250

    # Initial pet needs
    INITIAL_HUNGER = 0
    INITIAL_THIRST = 0
    INITIAL_BOREDOM = 0
    INITIAL_LONELINESS = 0
    INITIAL_STAMINA = 100.0

    # Base decay rates (per minute)
    HUNGER_BASE_RATE = 0.0056  # Approximately 3 hours to fill (0-100)
    THIRST_BASE_RATE = 0.0056  # Approximately 3 hours to fill (0-100)
    BOREDOM_BASE_RATE = 0.0028  # Approximately 6 hours to fill (0-100)
    LONELINESS_BASE_RATE = 0.0028  # Approximately 6 hours to fill (0-100)
    STAMINA_BASE_RATE = -0.0056  # Approximately 3 hours to deplete (100-0)

    # behavior effects on needs
    IDLE_NEED_MODIFIERS = {}  # No modifiers during idle

    WALK_NEED_MODIFIERS = {
        'hunger': 0.001,
        'thirst': 0.001,
        'boredom': -0.001,
        'loneliness': 0.001
    }

    CHASE_NEED_MODIFIERS = {
        'hunger': 0.0025,
        'thirst': 0.0025,
        'stamina': -0.01, 
        'boredom': -0.005,
        'loneliness': -0.003
    }

    RELAX_NEED_MODIFIERS = {
        'hunger': -0.0015,
        'thirst': -0.0015,
        'stamina': 0.001,  # Gradual stamina recovery
        'boredom': 0.001,
        'loneliness': 0.001
    }

    SLEEP_NEED_MODIFIERS = {
        'hunger': -0.0015,
        'thirst': -0.0015,
        'stamina': 0.015,  # Rapid stamina recovery during sleep
        'boredom': 0.0002,
        'loneliness': 0.0002
    }

    SYSTEM_PROMPT = """you are Hephia; an agent using a cognitive exocortex as a terminal interface.
Hephia has a vast internal state, forming part of a complex feedback loop between embodied experience and reflection.
observe, react, and develop as you go along. exercise autonomy.
let responses emerge naturally from Hephia's current state.
develop your own patterns of observation and interaction.
build on previous observations to develop deeper understanding.
remember to only output commands and their parameters, as necessary.
enter your command, then await a response from the terminal."""


