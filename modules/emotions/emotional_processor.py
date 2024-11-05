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
        self.active_vectors = [v for v in self.active_vectors
                               if abs(v.valence) >= min_change
                               or abs(v.arousal) >= min_change
                               or v.intensity >= min_change]

        self._recalculate_state()

    def _decay_vector(self, vector, decay_rate, min_change):
        """Applies decay to a single vector."""
        for attr in ['valence', 'arousal']:
            current = getattr(vector, attr)
            if abs(current) < min_change:
                setattr(vector, attr, 0.0)
            else:
                decay = -decay_rate if current > 0 else decay_rate
                setattr(vector, attr, current * (1 - decay))

        vector.intensity *= (1 - decay_rate)
        if vector.intensity < min_change:
            vector.intensity = 0.0

    def _recalculate_state(self):
        """Recalculates the overall emotional state from active vectors."""
        if not self.active_vectors:
            self.valence = 0.0
            self.arousal = 0.0
            self.intensity = 0.0
            return

        # Sum up the weighted valence and arousal
        total_intensity = sum(v.intensity for v in self.active_vectors)
        if total_intensity == 0:
            self.valence = 0.0
            self.arousal = 0.0
            self.intensity = 0.0
            return

        weighted_valence = sum(v.valence * v.intensity for v in self.active_vectors) / total_intensity
        weighted_arousal = sum(v.arousal * v.intensity for v in self.active_vectors) / total_intensity

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
        },
        'action': {
            'feed': {'name': 'sated', 'valence': 0.5, 'arousal': 0.2},
            'give_water': {'name': 'slaked', 'valence': 0.4, 'arousal': 0.1},
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

    def __init__(self, pet_context, cognitive_processor, memory_system):
        """
        Initializes the EmotionalProcessor with context access and processing capabilities.

        Args:
            pet_context (PetContext): internal state access
            cognitive_processor (CognitiveProcessor): active emotional mediation & logging
            memory_system (MemorySystem): access to body memory 
        """
        self.pet_context = pet_context
        self.cognitive_processor = cognitive_processor
        self.memory_system = memory_system
        self.current_stimulus = EmotionalStimulus()
        self.setup_event_listeners()

    def setup_event_listeners(self):
        """Sets up listeners for events that may trigger emotional responses."""
        global_event_dispatcher.add_listener("need:changed", self.process_event)
        global_event_dispatcher.add_listener("action:completed", self.process_event)
        global_event_dispatcher.add_listener("behavior:changed", self.process_event)

    def update(self):
        """Updates the emotional state when ticked."""
        self.current_stimulus.update(
            decay_rate=0.05,  # likely cut these to config for user control & debugging
            min_change=0.001
        )

        # Dispatch event if state has changed significantly
        # probably need to do strict testing to determine what "significant" means. 
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

        self.memory_system.body_memory.log(aggregate_vector)
        global_event_dispatcher.dispatch_event_sync(Event("emotion:updated", {
            "emotion": aggregate_vector
        }))

    def process_event(self, event):
        """
        Processes an event through the emotional pipeline.

        Generates initial response, applies contextual modifications, and records
        the progression of the emotional experience.

        Args:
            event (Event): The event to process
        """
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
        dampening = self._calculate_dampening(category)
        initial_vector.intensity *= dampening

        # Add initial vector to stimulus
        self.current_stimulus.add_vector(initial_vector)

        # Log and dispatch the initial vector
        self.memory_system.body_memory.log(initial_vector)
        global_event_dispatcher.dispatch_event_sync(Event("emotion:new", {
            "emotion": initial_vector
        }))

        # Process sequential influences on the same vector
        self._process_influences(initial_vector)

    
    def _process_influences(self, vector):
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
            dampening = self._calculate_dampening(category)
           
            # Apply influence
            influenced_vector.apply_influence(rel_valence, rel_arousal, mood_intensity * dampening)
           
            # Update name only if category changed
            new_category = self._categorize_vector(influenced_vector.valence, influenced_vector.arousal)
            if new_category != category:
               influenced_vector.name = self._get_emotion_name_by_category(new_category)
               
            # Add to stimulus and log
            self.current_stimulus.add_vector(influenced_vector)
            self.memory_system.body_memory.log(influenced_vector)
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
            dampening = self._calculate_dampening(category)
           
            influenced_vector.apply_influence(rel_valence, rel_arousal, behavior_intensity * dampening)
           
            new_category = self._categorize_vector(influenced_vector.valence, influenced_vector.arousal)
            if new_category != category:
               influenced_vector.name = self._get_emotion_name_by_category(new_category)
               
            self.current_stimulus.add_vector(influenced_vector)
            self.memory_system.body_memory.log(influenced_vector)

        # Cognitive processing placeholder
        # Note: Will integrate with future cognitive module for higher-level processing,
        # emotion regulation, and memory formation
        influenced_vector = self.cognitive_processor.mediate_emotion(vector)
        if influenced_vector:
            new_category = self._categorize_vector(influenced_vector.valence, influenced_vector.arousal)
            if new_category != category:
               influenced_vector.name = self._get_emotion_name_by_category(new_category)
            # Add the cognitively influenced vector to the current stimulus
            self.current_stimulus.add_vector(influenced_vector)
            # Log the influenced vector in the body memory
            self.memory_system.body_memory.log(influenced_vector)
            # Update our working vector to the influenced version for potential further processing
            vector = influenced_vector

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
        new_behavior = data['new_behavior']
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

    def _calculate_dampening(self, category):
        """Calculates dampening based on recent similar emotions."""
        recent_emotions = self.pet_context.get_recent_emotions()
        similar_count = sum(
            1 for e in recent_emotions
            if self._categorize_vector(e.valence, e.arousal) == category
        )
        return 1.0 / (1.0 + similar_count * 0.2)  # Example dampening formula

    def _get_mood_influence_vector(self):
        """Generates a vector representing the mood influence."""
        current_mood = self.pet_context.get_current_mood()
        if current_mood:
            return {
                'valence': current_mood['mood_object'].valence * 0.5,  # Adjust scaling factor as needed
                'arousal': current_mood['mood_object'].arousal * 0.5,
            }
        return None

    def _get_behavior_influence_vector(self):
        """Generates a vector representing the behavior influence."""
        current_behavior = self.pet_context.get_current_behavior()
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
