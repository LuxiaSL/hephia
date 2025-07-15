# modules/emotions/emotional_processor.py

"""
Emotional Processing Pipeline:
- Event triggers emotional response {immediate reaction} -> {models natural instinct}
- Response modified by current mood {colors perception} -> {affects emotional intensity}
- Behavior state influences emotion {physical state affects feelings} -> {models embodied cognition}
- Future cognitive mediation {thoughtful processing} -> {enables emotional growth}

Vectors accumulate and decay naturally, simulating the ebb and flow of emotional experience.
"""

from event_dispatcher import global_event_dispatcher, Event
from loggers import InternalLogger
from ..cognition.cognitive_bridge import CognitiveBridge
from ...internal_context import InternalContext
import time
import copy

class EmotionalVector:
    """Represents an individual emotional influence with magnitude, direction, and source tracking."""
    def __init__(self, valence, arousal, intensity, source_type=None, source_data=None, name=None):
        self.valence = valence
        self.arousal = arousal
        self.intensity = intensity
        self.source_type = source_type  # e.g., 'need', 'action', 'behavior'
        self.source_data = source_data  # original event data
        self.name = name
        self.timestamp = time.time()

    def to_dict(self):
        """Serializes EmotionalVector to dictionary."""
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'intensity': self.intensity,
            'source_type': self.source_type,
            'source_data': self.source_data,
            'name': self.name,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data):
        """Creates EmotionalVector from dictionary."""
        vector = cls(
            valence=data['valence'],
            arousal=data['arousal'],
            intensity=data['intensity'],
            source_type=data.get('source_type'),
            source_data=data.get('source_data'),
            name=data.get('name')
        )
        vector.timestamp = data.get('timestamp', time.time())
        return vector

    def __repr__(self):
        return (f"EmotionalVector(name={self.name}, valence={self.valence:.2f}, "
                f"arousal={self.arousal:.2f}, intensity={self.intensity:.2f}, "
                f"source={self.source_type})")

    def apply_influence(self, valence_delta, arousal_delta, intensity_delta):
        """Applies changes to the vector based on influence."""
        self.valence += valence_delta
        self.arousal += arousal_delta
        self.intensity += intensity_delta

        # Ensure values stay within bounds
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(-1.0, min(1.0, self.arousal))
        self.intensity = max(0.0, min(1.0, self.intensity))


class EmotionalStimulus:
    """
    Represents a continuous emotional state influenced by multiple emotional vectors.

    Maintains and manages a collection of active emotional influences, creating a
    realistic simulation of emotional flow where multiple feelings can coexist,
    interact, and naturally decay over time.
    """

    def __init__(self):
        self.active_vectors = []
        self.valence = 0.0
        self.arousal = 0.0
        self.intensity = 0.0

    def add_vector(self, vector):
        """Adds a new emotional vector to the active set."""
        self.active_vectors.append(vector)
        self._recalculate_state()

    def update(self, decay_rate, min_change):
        """Updates all active vectors and removes those that have decayed."""
        for vector in self.active_vectors:
            self._decay_vector(vector, decay_rate, min_change)

        # Remove fully decayed vectors
        self.active_vectors = [
            v for v in self.active_vectors
            if (abs(v.valence) >= min_change or
                abs(v.arousal) >= min_change or
                v.intensity >= min_change)
        ]

        self._recalculate_state()

    def _decay_vector(self, vector, decay_rate, min_change):
        """Applies decay to a single vector."""
        for attr in ['valence', 'arousal']:
            current = getattr(vector, attr)
            if abs(current) < min_change:
                setattr(vector, attr, 0.0)
            else:
                try:
                    # Decay magnitude toward zero regardless of sign
                    new_value = current * (1 - decay_rate)
                    # Bound values to prevent overflow
                    new_value = max(-1.0, min(1.0, new_value))
                    setattr(vector, attr, new_value)
                except (OverflowError, FloatingPointError):
                    # If overflow occurs, gracefully decay to 0
                    setattr(vector, attr, 0.0)


        # Handle intensity decay similarly
        try:
            vector.intensity *= (1 - decay_rate)
            vector.intensity = max(0.0, min(1.0, vector.intensity))
        except (OverflowError, FloatingPointError):
            vector.intensity = 0.0
            
        if vector.intensity < min_change:
            vector.intensity = 0.0

    def _recalculate_state(self):
        """Recalculates the overall emotional state from active vectors."""
        if not self.active_vectors:
            self.valence = 0.0
            self.arousal = 0.0
            self.intensity = 0.0
            return

        # Filter out invalid vectors and get total intensity
        valid_vectors = [v for v in self.active_vectors 
                        if v.intensity >= 0.0 and 
                        -1.0 <= v.valence <= 1.0 and 
                        -1.0 <= v.arousal <= 1.0]
        
        total_intensity = sum(v.intensity for v in valid_vectors)
        
        if total_intensity <= 0.0:
            self.valence = 0.0
            self.arousal = 0.0
            self.intensity = 0.0
            return

        try:
            # Calculate weighted values with validated inputs
            weighted_valence = 0.0
            weighted_arousal = 0.0
            
            # Calculate sums while checking for potential numerical issues
            for v in valid_vectors:
                try:
                    # Ensure intensity is valid for division
                    if not 0 <= v.intensity <= 1.0:
                        continue
                        
                    weighted_valence += v.valence * v.intensity
                    weighted_arousal += v.arousal * v.intensity
                except (OverflowError, FloatingPointError):
                    continue
                    
            # Only perform division if we have valid total intensity
            if total_intensity > 0:
                weighted_valence /= total_intensity
                weighted_arousal /= total_intensity
            else:
                weighted_valence = 0.0
                weighted_arousal = 0.0
                
        except (OverflowError, FloatingPointError, ZeroDivisionError):
            # Fallback to neutral state if calculations fail
            weighted_valence = 0.0 
            weighted_arousal = 0.0

        # Normalize and bound values
        self.valence = max(-1.0, min(1.0, weighted_valence))
        self.arousal = max(-1.0, min(1.0, weighted_arousal))
        self.intensity = max(0.0, min(1.0, total_intensity))


class EmotionalProcessor:
    """
    Processes events and generates appropriate emotional responses.

    Maintains a continuous emotional state that can be influenced by various stimuli,
    creating a naturalistic flow of emotional experience. Responses are generated
    based on events, modified by current state and context, and recorded in memory.
    """

    # Emotional categories for dampening and processing
    EMOTION_CATEGORIES = {
        'joy': {'valence_range': (0.5, 1.0), 'arousal_range': (0.5, 1.0)},
        'contentment': {'valence_range': (0.2, 0.5), 'arousal_range': (-0.5, 0.5)},
        'sadness': {'valence_range': (-1.0, -0.5), 'arousal_range': (-0.5, 0.5)},
        'frustration': {'valence_range': (-0.5, 0.0), 'arousal_range': (0.0, 0.5)},
        'calm': {'valence_range': (0.0, 0.5), 'arousal_range': (-1.0, -0.5)},
        'anxiety': {'valence_range': (-0.5, 0.0), 'arousal_range': (0.5, 1.0)},
        # Add more categories as needed
    }

    # Base emotional mappings for different stimuli
    EMOTION_MAPPINGS = {
        'need': {
            'hunger': {
                'increase': {'name': 'starving', 'valence': -0.4, 'arousal': 0.3},
                'decrease': {'name': 'sated', 'valence': 0.4, 'arousal': -0.2}
            },
            'thirst': {
                'increase': {'name': 'parched', 'valence': -0.4, 'arousal': 0.3},
                'decrease': {'name': 'slaked', 'valence': 0.4, 'arousal': -0.2}
            },
            'boredom': {
                'increase': {'name': 'bored', 'valence': -0.3, 'arousal': 0.2},
                'decrease': {'name': 'curious', 'valence': 0.3, 'arousal': 0.2}
            },
            'stamina': {
                'increase': {'name': 'energetic', 'valence': 0.3, 'arousal': 0.5},
                'decrease': {'name': 'tired', 'valence': -0.3, 'arousal': -0.4}
            },
            'loneliness':{
                'increase': {'name': 'forlorn', 'valence': -0.3, 'arousal': -0.5},
                'decrease': {'name': 'connected', 'valence': 0.3, 'arousal': 0.4}
            }
        },
        'action': {
            'feed': {'name': 'sated', 'valence': 0.5, 'arousal': 0.2},
            'drink': {'name': 'slaked', 'valence': 0.4, 'arousal': 0.1},
            'play': {'name': 'joyful', 'valence': 0.7, 'arousal': 0.8},
            'rest': {'name': 'relaxed', 'valence': 0.3, 'arousal': -0.4}
        },
        'behavior': {
            'idle': {'name': 'calm', 'valence': 0.1, 'arousal': -0.2},
            'walk': {'name': 'alert', 'valence': 0.4, 'arousal': 0.5},
            'chase': {'name': 'excited', 'valence': 0.7, 'arousal': 0.8},
            'sleep': {'name': 'peaceful', 'valence': 0.1, 'arousal': -0.8},
            'relax': {'name': 'content', 'valence': 0.3, 'arousal': -0.4}
            # Add more behaviors as they are implemented
        }
    }

    def __init__(self, internal_context: InternalContext, cognitive_bridge: CognitiveBridge) -> None:
        """
        Initializes the EmotionalProcessor with context access and processing capabilities.

        Args:
            internal_context (InternalContext): internal state access
            cognitive_bridge (CognitiveBridge): active emotional mediation & logging
        """
        self.internal_context = internal_context
        self.cognitive_bridge = cognitive_bridge
        self.current_stimulus = EmotionalStimulus()
        self.setup_event_listeners()

    def setup_event_listeners(self):
        """Sets up listeners for events that may trigger emotional responses."""
        global_event_dispatcher.add_listener("need:changed", self.process_event)
        global_event_dispatcher.add_listener("action:completed", self.process_event)
        global_event_dispatcher.add_listener("behavior:changed", self.process_event)

        global_event_dispatcher.add_listener("memory:echo", self._handle_memory_echo)
        global_event_dispatcher.add_listener("cognitive:emotional:meditation", self.process_meditation)
        global_event_dispatcher.add_listener("cognitive:emotional:influence", self.process_cognitive_influence)

    def update(self):
        """Updates the emotional state when ticked."""
        self.current_stimulus.update(
            decay_rate=0.05,  # likely cut these to config for user control & debugging
            min_change=0.001
        )

        # Dispatch event if state has changed significantly
        # probably need to do strict testing to determine what "significant" means. (2025 luxia: oh if only you knew...)
        # number testing here will be important; gonna need users and people to play around with everything to give me feedback.
        if (abs(self.current_stimulus.valence) > 0.01 or
            abs(self.current_stimulus.arousal) > 0.01 or
            self.current_stimulus.intensity > 0.01):
            self._dispatch_current_state()
 
    def _dispatch_current_state(self):
        """Dispatches the current emotional state."""
        category = self._categorize_stimulus(self.current_stimulus)
        emotion_name = self._get_emotion_name_by_category(category)

        aggregate_vector = EmotionalVector(
            valence=self.current_stimulus.valence,
            arousal=self.current_stimulus.arousal,
            intensity=self.current_stimulus.intensity,
            name=emotion_name,
            source_type='aggregate'
        )

        global_event_dispatcher.dispatch_event(Event("emotion:updated", {
            "emotion": aggregate_vector
        }))

    async def process_event(self, event: Event):
        """
        Processes an event through the emotional pipeline.

        Generates initial response, applies contextual modifications, and records
        the progression of the emotional experience.

        Args:
            event (Event): The event to process
        """
        if event.event_type == "cognitive:emotional:influence":
            # Skip processing if this is a cognitive influence event
            return
        
        # Generate initial emotional vector
        vector_data, name = self._generate_emotional_vector(event.event_type, event.data)
        if not vector_data:
            return

        # Create the initial emotional vector
        initial_vector = EmotionalVector(
            valence=vector_data['valence'],
            arousal=vector_data['arousal'],
            intensity=self._calculate_intensity('event', vector_data),
            source_type=event.event_type,
            source_data=event.data,
            name=name
        )

        # Apply dampening
        category = self._categorize_vector(initial_vector.valence, initial_vector.arousal)
        dampening = await self._calculate_dampening(category)
        initial_vector.intensity *= dampening

        # Add initial vector to stimulus
        self.current_stimulus.add_vector(initial_vector)

        # Dispatch the initial vector
        global_event_dispatcher.dispatch_event(Event("emotion:new", {
            "emotion": initial_vector
        }))

        # Process sequential influences on the same vector
        await self._process_influences(initial_vector)

        # After processing all influences, dispatch final aggregate state
        # Recalculate overall emotional state 
        category = self._categorize_stimulus(self.current_stimulus)
        emotion_name = self._get_emotion_name_by_category(category)
        
        final_state = EmotionalVector(
            valence=self.current_stimulus.valence,
            arousal=self.current_stimulus.arousal,
            intensity=self.current_stimulus.intensity,
            name=emotion_name,
            source_type='aggregate',
            source_data={'initial_trigger': initial_vector.source_data}
        )

        global_event_dispatcher.dispatch_event(Event("emotion:finished", {
            "emotion": final_state,
            "initial": initial_vector
        }))

    
    async def _process_influences(self, vector: EmotionalVector):
        """
        Processes mood and behavior influences on the given vector.
       
        Each influence is calculated relative to the current vector's state,
        simulating how current emotions color our perception of new stimuli.
        """
        # Apply mood influence
        mood_vector_data = self._get_mood_influence_vector()
        if mood_vector_data:
            # Copy while maintaining original event data
            influenced_vector = copy.deepcopy(vector)
           
            # Calculate relative influence
            rel_valence, rel_arousal = self._calculate_relative_influence(
                vector, 
                mood_vector_data['valence'],
                mood_vector_data['arousal'],
                influence_type='mood'
            )
           
            # Calculate and apply dampening
            mood_intensity = self._calculate_intensity('mood', mood_vector_data)
            category = self._categorize_vector(influenced_vector.valence, influenced_vector.arousal)
            dampening = await self._calculate_dampening(category)
           
            # Apply influence
            influenced_vector.apply_influence(rel_valence, rel_arousal, mood_intensity * dampening)
           
            # Update name only if category changed
            new_category = self._categorize_vector(influenced_vector.valence, influenced_vector.arousal)
            if new_category != category:
               influenced_vector.name = self._get_emotion_name_by_category(new_category)
               
            # Add to stimulus and log
            self.current_stimulus.add_vector(influenced_vector)
            vector = influenced_vector  # Continue with influenced vector

        # Apply behavior influence similarly
        behavior_vector_data = self._get_behavior_influence_vector()
        if behavior_vector_data:
            influenced_vector = copy.deepcopy(vector)
           
            rel_valence, rel_arousal = self._calculate_relative_influence(
               vector,
               behavior_vector_data['valence'],
               behavior_vector_data['arousal'],
               influence_type='behavior'
            )
           
            behavior_intensity = self._calculate_intensity('behavior', behavior_vector_data)
            category = self._categorize_vector(influenced_vector.valence, influenced_vector.arousal)
            dampening = await self._calculate_dampening(category)
           
            influenced_vector.apply_influence(rel_valence, rel_arousal, behavior_intensity * dampening)
           
            new_category = self._categorize_vector(influenced_vector.valence, influenced_vector.arousal)
            if new_category != category:
               influenced_vector.name = self._get_emotion_name_by_category(new_category)
               
            self.current_stimulus.add_vector(influenced_vector)

        # TODO: apply memory content influence
        # to come: query across memory nodes for average network emotional resonance to influence vector by
        # just as well, the direct cognitive influence via mediate_emotion and one_turn call. could be done via an event which then gets listened to here and files back in after? want to make sure we don't start a cycle.

    def _calculate_relative_influence(self, base_vector, influence_valence, influence_arousal, influence_type):
        """
        Calculates how much an influence affects the current emotional state.

        Args:
           base_vector (EmotionalVector): Current emotional vector
           influence_valence (float): Raw valence influence
           influence_arousal (float): Raw arousal influence
           influence_type (str): Type of influence ('mood' or 'behavior')

        Returns:
           tuple: (relative_valence, relative_arousal)
        """
        # Base factors for different influence types
        influence_factors = {
           'mood': 0.5,    # Mood has moderate influence
           'behavior': 0.3  # Behavior has lesser influence
        }
        base_factor = influence_factors.get(influence_type, 0.1)
       
        # Calculate resistance based on current state
        # Stronger emotions are harder to influence
        state_resistance = abs(base_vector.valence) + abs(base_vector.arousal)
       
        # Calculate relative changes
        # If current valence and influence valence have opposite signs,
        # the influence is reduced (harder to make happy when sad)
        valence_agreement = 1.0 if (base_vector.valence * influence_valence) >= 0 else 0.5
        arousal_agreement = 1.0 if (base_vector.arousal * influence_arousal) >= 0 else 0.7
       
        relative_valence = influence_valence * base_factor * valence_agreement / (1 + state_resistance)
        relative_arousal = influence_arousal * base_factor * arousal_agreement / (1 + state_resistance)
       
        return relative_valence, relative_arousal

    def _generate_emotional_vector(self, event_type, event_data):
        """
        Generates the initial emotional vector based on event type and data.

        Args:
            event_type (str): Type of event
            event_data (dict): Event data

        Returns:
            tuple: (vector dict, emotion name) or (None, None) if no response
        """
        if event_type == "need:changed":
            return self._process_need_event(event_data)
        elif event_type == "action:completed":
            return self._process_action_event(event_data)
        elif event_type == "behavior:changed":
            return self._process_behavior_event(event_data)
        return None, None

    def _process_need_event(self, data):
        """Processes need-related events into emotional vectors."""
        need_name = data['need_name']
        old_value = data['old_value']
        new_value = data['new_value']
        change = new_value - old_value

        if abs(change) < 5:  # Ignore minor changes
            return None, None

        direction = 'increase' if change > 0 else 'decrease'
        mapping = self.EMOTION_MAPPINGS['need'].get(need_name, {}).get(direction)

        if mapping:
            return mapping, mapping['name']
        else:
            return None, None

    def _process_action_event(self, data):
        """Processes action-related events into emotional vectors."""
        action_name = data['action_name']
        mapping = self.EMOTION_MAPPINGS['action'].get(action_name)

        if mapping:
            return mapping, mapping['name']
        else:
            return None, None

    def _process_behavior_event(self, data):
        """Processes behavior-related events into emotional vectors."""
        new_behavior = data['new_name']
        mapping = self.EMOTION_MAPPINGS['behavior'].get(new_behavior)

        if mapping:
            return mapping, mapping['name']
        else:
            return None, None

    def _calculate_intensity(self, influence_source, vector):
        """Calculates intensity based on the source of the influence."""
        if influence_source == "event":
            # Intensity can be adjusted based on the magnitude of change
            return vector.get('intensity', 0.5)
        elif influence_source == "mood":
            return 0.3  # Adjust as needed
        elif influence_source == "behavior":
            return 0.2  # Adjust as needed
        return 0.1  # Default minimal intensity

    async def _calculate_dampening(self, category):
        """
        Calculates dampening based on recent similar emotions.
        
        Higher counts of similar recent emotions result in stronger dampening,
        simulating emotional fatigue/adaptation.
        
        Args:
            category (str): Emotional category to check for repetition
            
        Returns:
            float: Dampening factor between 0.0 and 1.0
        """
        recent_emotions = await self.internal_context.get_recent_emotions()
        similar_count = sum(
            1 for emotion in recent_emotions
            if emotion and 'valence' in emotion and 'arousal' in emotion and
            self._categorize_vector(
                float(emotion['valence']),
                float(emotion['arousal'])
            ) == category
        )
        
        # Same dampening formula, validated with strong emotions
        return 1.0 / (1.0 + similar_count * 0.2)  # Returns 1.0 to 0.2 range

    def _get_mood_influence_vector(self):
        """Generates a vector representing the mood influence."""
        current_mood = self.internal_context.get_current_mood()
        if current_mood:
            return {
                'valence': current_mood['mood_object'].valence * 0.5,
                'arousal': current_mood['mood_object'].arousal * 0.5,
            }
        return None

    def _get_behavior_influence_vector(self):
        """Generates a vector representing the behavior influence."""
        current_behavior = self.internal_context.get_current_behavior()
        behavior_name = current_behavior.name
        mapping = self.EMOTION_MAPPINGS['behavior'].get(behavior_name)

        if mapping:
            return mapping
        else:
            return None

    def _categorize_stimulus(self, stimulus):
        """Finds the category for the current emotional state."""
        return self._categorize_vector(stimulus.valence, stimulus.arousal)

    def _categorize_vector(self, valence, arousal):
        """Categorizes an emotion based on valence and arousal."""
        for category, ranges in self.EMOTION_CATEGORIES.items():
            if (ranges['valence_range'][0] <= valence <= ranges['valence_range'][1] and
                    ranges['arousal_range'][0] <= arousal <= ranges['arousal_range'][1]):
                return category
        return 'neutral'

    def _get_emotion_name_by_category(self, category):
        """Converts a category to an emotion name."""
        category_to_emotion = {
            'joy': 'joyful',
            'contentment': 'content',
            'sadness': 'sad',
            'frustration': 'frustrated',
            'calm': 'calm',
            'anxiety': 'anxious',
            'neutral': 'neutral'
            # Add more mappings as needed
        }
        return category_to_emotion.get(category, 'neutral')

    def get_emotional_state(self):
        """
        retrieve the exact bare minimum info to maintain continuity between sessions for this module
        i.e. the active vectors
        """
        return {
            "active_vectors": [
                {
                    "valence": vector.valence,
                    "arousal": vector.arousal,
                    "intensity": vector.intensity,
                    "name": vector.name,
                    "source_type": vector.source_type,
                    "timestamp": vector.timestamp,
                    **({"source_data": vector.source_data} 
                       if isinstance(vector.source_data, (dict, str, int, float, bool)) 
                       else {})
                }
                for vector in self.current_stimulus.active_vectors
                if (abs(vector.valence) >= 0.001 or 
                    abs(vector.arousal) >= 0.001 or 
                    vector.intensity >= 0.001)
            ]
        }

    def set_emotional_state(self, state):
        """
        restore from prior session
        """
        if not state:
            return
            
        self.current_stimulus = EmotionalStimulus()
        
        for vector_data in state.get("active_vectors", []):
            vector = EmotionalVector(
                valence=vector_data["valence"],
                arousal=vector_data["arousal"],
                intensity=vector_data["intensity"],
                name=vector_data.get("name"),
                source_type=vector_data.get("source_type")
            )
            if "timestamp" in vector_data:
                vector.timestamp = float(vector_data["timestamp"])
            if "source_data" in vector_data:
                vector.source_data = vector_data["source_data"]
            
            self.current_stimulus.add_vector(vector)

    async def _handle_memory_echo(self, event):
        """
        Processes memory echo events into emotional influences.
        Memory echoes are resonance effects from recalled emotional states.
        
        Args:
            event (Event): The memory echo event containing emotional metadata
        """
        echo_data = event.data
        if not echo_data or 'metadata' not in echo_data:
            return
            
        # Extract metadata and check for mood since emotional array is empty
        metadata = echo_data['metadata']
        intensity = echo_data.get('intensity', 0.3)

        # First try emotional data
        if 'emotional' in metadata and metadata['emotional']:
            primary_emotion = metadata['emotional'][0]
            emotion_source = primary_emotion
        # Fall back to mood data if available
        elif 'mood' in metadata and metadata['mood'].get('mood'):
            emotion_source = metadata['mood']['mood']
        else:
            return
            
        # Create echo vector from emotion source
        echo_vector = EmotionalVector(
            valence=emotion_source['valence'],
            arousal=emotion_source['arousal'], 
            intensity=intensity * emotion_source.get('intensity', 0.25),
            source_type='memory_echo',
            source_data={'echo_source': echo_data.get('source_node')},
            name=emotion_source.get('name', 'echo')
        )
        
        # Apply echo-specific dampening
        category = self._categorize_vector(echo_vector.valence, echo_vector.arousal)
        echo_dampening = await self._calculate_dampening(category) * 0.7  # Echo specific reduction
        echo_vector.intensity *= echo_dampening
        
        # Add echo vector to current stimulus
        self.current_stimulus.add_vector(echo_vector)
        
        # Log and dispatch the echo vector
        global_event_dispatcher.dispatch_event_sync(Event("emotion:echo", {
            "emotion": echo_vector,
            "source": echo_data.get('source_node')
        }))
        
        # Process influences on echo vector
        await self._process_influences(echo_vector)

    async def process_meditation(self, event):
        """Processes meditation events for emotional influence."""
        try:
            meditation_data = event.data
            meditation_type = meditation_data.get('type')
            intensity = meditation_data.get('intensity', 0.5)
            duration = meditation_data.get('duration', 1)

            # Map meditation types to emotional vectors
            meditation_mappings = {
                'calming': {'valence': 0.3, 'arousal': -0.6, 'name': 'peaceful'},
                'focusing': {'valence': 0.2, 'arousal': 0.4, 'name': 'focused'},
                'satisfaction': {'valence': 0.4, 'arousal': -0.2, 'name': 'content'},
                'activation': {'valence': 0.3, 'arousal': 0.5, 'name': 'energized'},
                'memory_informed': None  # Special handling below
            }

            if meditation_type == 'memory_informed':
                # Handle memory-informed meditation with discovered emotional patterns
                state_query = meditation_data.get('state_query', 'focused')
                valence_direction = meditation_data.get('valence_direction', 0.0)
                arousal_direction = meditation_data.get('arousal_direction', 0.0)
                memory_count = meditation_data.get('memory_count', 0)
                
                # Create vector based on discovered emotional patterns
                meditation_vector = EmotionalVector(
                    valence=valence_direction * 0.7,  # Scale down the influence
                    arousal=arousal_direction * 0.7,
                    intensity=intensity * (1 + 0.1 * memory_count),  # Boost intensity based on memory count
                    source_type='memory_informed_meditation',
                    source_data={
                        'state_query': state_query,
                        'memory_count': memory_count,
                        'duration': duration
                    },
                    name=f"memory-informed {state_query}"
                )
                
            elif meditation_type in meditation_mappings:
                mapping = meditation_mappings[meditation_type]
                
                # Create meditation vector
                meditation_vector = EmotionalVector(
                    valence=mapping['valence'],
                    arousal=mapping['arousal'],
                    intensity=intensity * (1 + 0.2 * duration),  # Intensity increases with duration
                    source_type='meditation',
                    source_data={'type': meditation_type, 'duration': duration},
                    name=mapping['name']
                )
            else:
                return  # Unknown meditation type

            # Apply meditation-specific dampening
            category = self._categorize_vector(meditation_vector.valence, meditation_vector.arousal)
            meditation_dampening = await self._calculate_dampening(category) * 0.8
            meditation_vector.intensity *= meditation_dampening

            # Add meditation vector to current stimulus
            self.current_stimulus.add_vector(meditation_vector)

            # Log and dispatch meditation effect (counts as new emotion for systems)
            global_event_dispatcher.dispatch_event_sync(Event("emotion:new", {
                "emotion": meditation_vector,
                "duration": duration
            }))

            # Process influences on meditation vector
            await self._process_influences(meditation_vector)

        except Exception as e:
            print(f"Error processing meditation: {str(e)}")

    async def process_cognitive_influence(self, event: Event):
        """
        Process cognitive influence events into emotional responses.
        
        Cognitive influences represent the emotional impact of thoughts and mental processes.
        These are applied directly as EmotionalVectors to influence current emotional state.
        
        Args:
            event (Event): Event containing list of influence dictionaries
        """
        try:
            influences_data = event.data.get('influences', [])
            if not influences_data:
                InternalLogger.warning("Cognitive influence event received with no influences")
                return

            InternalLogger.info(f"Processing {len(influences_data)} cognitive influences")

            applied_influences = []
            
            for influence_data in influences_data:
                try:
                    # Create EmotionalVector from influence data
                    cognitive_vector = EmotionalVector(
                        valence=influence_data.get('valence', 0.0),
                        arousal=influence_data.get('arousal', 0.0),
                        intensity=influence_data.get('intensity', 0.0),
                        source_type='cognitive_influence',
                        source_data=influence_data.get('source_data', {}),
                        name=influence_data.get('name', 'cognitive')
                    )
                    
                    # Apply cognitive-specific dampening
                    category = self._categorize_vector(cognitive_vector.valence, cognitive_vector.arousal)
                    cognitive_dampening = await self._calculate_dampening(category) * 0.9
                    cognitive_vector.intensity *= cognitive_dampening
                    
                    # Skip very weak influences after dampening
                    if cognitive_vector.intensity < 0.02:
                        continue
                    
                    # Add to current stimulus
                    self.current_stimulus.add_vector(cognitive_vector)
                    applied_influences.append(cognitive_vector)
                    
                    # Dispatch individual cognitive influence event for logging/monitoring
                    global_event_dispatcher.dispatch_event_sync(Event("emotion:cognitive", {
                        "emotion": cognitive_vector,
                        "source": influence_data.get('source_data', {}).get('interface', 'unknown'),
                        "analysis_method": influence_data.get('source_data', {}).get('analysis_method', 'unknown')
                    }))
                    
                except Exception as influence_error:
                    InternalLogger.error(f"Error processing individual cognitive influence: {influence_error}")
                    continue
            
            if applied_influences:
                InternalLogger.info(f"Applied {len(applied_influences)} cognitive influences to emotional state")

                # Process sequential influences on the applied vectors
                for vector in applied_influences:
                    await self._process_influences(vector)
                
                # Dispatch final cognitive influence completion event
                final_state = EmotionalVector(
                    valence=self.current_stimulus.valence,
                    arousal=self.current_stimulus.arousal,
                    intensity=self.current_stimulus.intensity,
                    name=self._get_emotion_name_by_category(
                        self._categorize_stimulus(self.current_stimulus)
                    ),
                    source_type='cognitive_aggregate',
                    source_data={
                        'applied_count': len(applied_influences),
                        'source_interface': event.data.get('source_interface', 'unknown')
                    }
                )
                
                global_event_dispatcher.dispatch_event_sync(Event("emotion:cognitive_complete", {
                    "emotion": final_state,
                    "applied_influences": len(applied_influences),
                    "source_interface": event.data.get('source_interface', 'unknown')
                }))
            else:
                InternalLogger.debug("No cognitive influences were strong enough to apply after dampening")

        except Exception as e:
            InternalLogger.error(f"Error processing cognitive influence event: {e}")
            # Ensure we don't crash the emotional system on cognitive influence errors
