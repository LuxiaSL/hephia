"""
node_utils.py

Shared utility functions for memory node operations. Provides common calculations,
state management helpers, and connection utilities used by both body and cognitive
memory nodes.

Focuses on:
- Connection weight calculations
- State comparison utilities
- Cross-platform serialization helpers
- Common node operations
"""

import math
import json
from typing import Dict, List, Any, Optional

class NodeUtilsError(Exception):
    """Base exception for node utility errors."""
    pass

# -------------------------------------------------------------------------
# Connection & Weight Utilities
# -------------------------------------------------------------------------
def softmax(scores: List[float]) -> List[float]:
    """
    Calculate softmax of input scores with numerical stability.
    Used for connection weight normalization.
    
    Args:
        scores: List of raw scores to normalize
        
    Returns:
        List of normalized probabilities summing to 1
    """
    if not scores:
        return []
        
    # Shift for numerical stability
    shifted = [s - max(scores) for s in scores]
    exp_scores = [math.exp(s) for s in shifted]
    sum_exp = sum(exp_scores)
    
    return [e / sum_exp for e in exp_scores]

def calculate_temporal_weight(
    timestamp1: float,
    timestamp2: float,
    decay_hours: float = 24.0
) -> float:
    """
    Calculate connection weight modifier based on temporal proximity.
    
    Args:
        timestamp1, timestamp2: Timestamps to compare
        decay_hours: Hours for full decay
        
    Returns:
        float: Weight modifier [0-1]
    """
    time_diff = abs(timestamp1 - timestamp2)
    decay_seconds = decay_hours * 3600
    return math.exp(-time_diff / decay_seconds)

def normalize_connections(
    connections: Dict[str, float],
    min_weight: float = 0.2
) -> Dict[str, float]:
    """
    Normalize connection weights and prune weak connections.
    
    Args:
        connections: Dict of node_id to weight mappings
        min_weight: Minimum weight to retain
        
    Returns:
        Dict of normalized connections
    """
    if not connections:
        return {}
        
    # Filter weak connections
    strong = {k: v for k, v in connections.items() if v >= min_weight}
    
    if not strong:
        return {}
        
    # Normalize remaining weights
    total = sum(strong.values())
    return {k: v/total for k, v in strong.items()}

# -------------------------------------------------------------------------
# State Comparison Utilities 
# -------------------------------------------------------------------------
def calculate_emotional_similarity(
    emotions1: List[Dict[str, Any]],
    emotions2: List[Dict[str, Any]]
) -> float:
    """
    Calculate similarity between emotional states.
    Handles both vector and categorical emotions.
    
    Args:
        emotions1, emotions2: Lists of emotion dictionaries
        
    Returns:
        float: Similarity score [0-1]
    """
    if not emotions1 or not emotions2:
        return 0.0
        
    similarities = []
    for e1 in emotions1:
        for e2 in emotions2:
            # Compare core emotional dimensions
            sim = (
                (1 - abs(e1.get('valence', 0) - e2.get('valence', 0)) * 0.4) +
                (1 - abs(e1.get('arousal', 0) - e2.get('arousal', 0)) * 0.4) +
                (1 - abs(e1.get('intensity', 0) - e2.get('intensity', 0)) * 0.2)
            )
            similarities.append(sim)
                
    return sum(similarities) / len(similarities) if similarities else 0.0

def calculate_needs_similarity(
    needs1: Dict[str, Dict[str, Any]],
    needs2: Dict[str, Dict[str, Any]]
) -> float:
    """
    Calculate similarity between need states.
    
    Args:
        needs1, needs2: Need state dictionaries
        
    Returns:
        float: Similarity score [0-1]
    """
    if not needs1 or not needs2:
        return 0.0
        
    common_needs = set(needs1.keys()) & set(needs2.keys())
    if not common_needs:
        return 0.0
        
    similarities = []
    for need in common_needs:
        val1 = needs1[need].get('value', 50)
        val2 = needs2[need].get('value', 50)
        similarities.append(1 - abs(val1 - val2) / 100)
        
    return sum(similarities) / len(similarities)

def calculate_mood_similarity(
    mood1: Dict[str, Any],
    mood2: Dict[str, Any]
) -> float:
    """
    Calculate similarity between mood states.
    
    Args:
        mood1, mood2: Mood state dictionaries
        
    Returns:
        float: Similarity score [0-1]
    """
    if not mood1 or not mood2:
        return 0.0
        
    sim = (
        (1 - abs(mood1.get('valence', 0) - mood2.get('valence', 0)) * 0.5) +
        (1 - abs(mood1.get('arousal', 0) - mood2.get('arousal', 0)) * 0.5)
    )
    return sim

# -------------------------------------------------------------------------
# Serialization Utilities
# -------------------------------------------------------------------------
def serialize_state(
    state: Dict[str, Any],
    include_timestamps: bool = True
) -> str:
    """
    Safely serialize state dictionary to JSON string.
    Handles datetime objects and special types.
    
    Args:
        state: State dictionary to serialize
        include_timestamps: Whether to include timestamps
        
    Returns:
        str: JSON serialized state
    """
    try:
        # Remove timestamps if not wanted
        if not include_timestamps:
            state = {k: v for k, v in state.items() if not k.endswith('_timestamp')}
            
        return json.dumps(state, default=str)
    except Exception as e:
        raise NodeUtilsError(f"Failed to serialize state: {e}")

def deserialize_state(state_str: str) -> Dict[str, Any]:
    """
    Safely deserialize state from JSON string.
    
    Args:
        state_str: JSON string to deserialize
        
    Returns:
        Dict: Deserialized state dictionary
    """
    try:
        return json.loads(state_str)
    except json.JSONDecodeError as e:
        raise NodeUtilsError(f"Failed to deserialize state: {e}")

# -------------------------------------------------------------------------
# Common Node Operations
# -------------------------------------------------------------------------
def blend_states(
    state1: Dict[str, Any],
    state2: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    is_raw: bool = True
) -> Dict[str, Any]:
    """
    Blend two state dictionaries with enhanced handling for raw/processed states
    and cognitive context. Used for merging nodes or creating synthetic states.
    
    Args:
        state1, state2: States to blend
        weights: Optional weight for each state component
        is_raw: Whether states are raw (True) or processed (False)
        
    Returns:
        Dict: Blended state dictionary
    """
    if weights is None:
        weights = {
            'emotional': 0.4,
            'needs': 0.3,
            'mood': 0.2,
            'cognitive': 0.1
        }
        
    blended = {}
    
    # Handle emotional state blending
    if is_raw:
        # Raw state emotional vectors
        if 'emotions' in state1 and 'emotions' in state2:
            vectors1 = state1['emotions'].get('active_vectors', [])
            vectors2 = state2['emotions'].get('active_vectors', [])
            w = weights.get('emotional', 0.4)
            
            # Combine unique vectors by name
            vector_map = {}
            for v in vectors1 + vectors2:
                name = v.get('name')
                if name not in vector_map:
                    vector_map[name] = v
                else:
                    # For duplicate vectors, blend or take strongest
                    existing = vector_map[name]
                    if abs(v.get('intensity', 0) - existing.get('intensity', 0)) > 0.3:
                        vector_map[name] = max([v, existing], 
                            key=lambda x: x.get('intensity', 0))
                    else:
                        vector_map[name] = {
                            'name': name,
                            'valence': existing['valence'] * w + v['valence'] * (1-w),
                            'arousal': existing['arousal'] * w + v['arousal'] * (1-w),
                            'intensity': existing['intensity'] * w + v['intensity'] * (1-w),
                            'source_type': existing.get('source_type')
                        }
                        
            blended['emotions'] = {'active_vectors': list(vector_map.values())}
            
    else:
        # Processed state emotional states
        if 'emotional_state' in state1 and 'emotional_state' in state2:
            emotions1 = state1['emotional_state']
            emotions2 = state2['emotional_state']
            w = weights.get('emotional', 0.4)
            
            # Handle overall emotional states (first element)
            blended_emotions = []
            if emotions1 and emotions2:
                overall1 = emotions1[0]
                overall2 = emotions2[0]
                blended_emotions.append({
                    'name': overall1['name'] if overall1['intensity'] >= overall2['intensity'] else overall2['name'],
                    'intensity': overall1['intensity'] * w + overall2['intensity'] * (1-w),
                    'valence': overall1['valence'] * w + overall2['valence'] * (1-w),
                    'arousal': overall1['arousal'] * w + overall2['arousal'] * (1-w)
                })
                
            # Combine remaining emotion vectors
            vector_map = {}
            for e in (emotions1[1:] if len(emotions1) > 1 else []) + (emotions2[1:] if len(emotions2) > 1 else []):
                if e['name'] not in vector_map:
                    vector_map[e['name']] = e
                else:
                    existing = vector_map[e['name']]
                    if abs(e['intensity'] - existing['intensity']) > 0.3:
                        vector_map[e['name']] = max([e, existing], key=lambda x: x['intensity'])
                    else:
                        vector_map[e['name']] = {
                            'name': e['name'],
                            'category': e['category'],
                            'intensity': e['intensity'] * w + existing['intensity'] * (1-w),
                            'source': e.get('source')
                        }
                        
            blended_emotions.extend(list(vector_map.values()))
            blended['emotional_state'] = blended_emotions

    # Blend needs states
    need_key = 'needs'
    if need_key in state1 and need_key in state2:
        needs1 = state1[need_key]
        needs2 = state2[need_key]
        w = weights.get('needs', 0.3)
        
        blended[need_key] = {}
        all_needs = set(needs1.keys()) | set(needs2.keys())
        
        for need in all_needs:
            if is_raw:
                n1 = needs1.get(need, {'value': 50, 'urgency': 0})
                n2 = needs2.get(need, {'value': 50, 'urgency': 0})
                urgency1 = n1.get('urgency', 0)
                urgency2 = n2.get('urgency', 0)
                
                if abs(urgency1 - urgency2) > 0.3:
                    blended[need_key][need] = n1 if urgency1 >= urgency2 else n2
                else:
                    blended[need_key][need] = {
                        'value': n1.get('value', 50) * w + n2.get('value', 50) * (1-w),
                        'urgency': urgency1 * w + urgency2 * (1-w),
                        'last_update': max(n1.get('last_update', 0), n2.get('last_update', 0))
                    }
            else:
                # For processed needs, just blend the values
                val1 = needs1.get(need, 0)
                val2 = needs2.get(need, 0)
                blended[need_key][need] = val1 * w + val2 * (1-w)

    # Blend mood states
    mood_key = 'mood'
    if mood_key in state1 and mood_key in state2:
        mood1 = state1[mood_key]
        mood2 = state2[mood_key]
        w = weights.get('mood', 0.2)
        
        if is_raw:
            blended[mood_key] = {
                'valence': mood1.get('valence', 0) * w + mood2.get('valence', 0) * (1-w),
                'arousal': mood1.get('arousal', 0) * w + mood2.get('arousal', 0) * (1-w),
                'last_update': max(mood1.get('last_update', 0), mood2.get('last_update', 0))
            }
        else:
            blended[mood_key] = {
                'name': mood1['name'] if mood1.get('valence', 0) >= mood2.get('valence', 0) else mood2['name'],
                'valence': mood1.get('valence', 0) * w + mood2.get('valence', 0) * (1-w),
                'arousal': mood1.get('arousal', 0) * w + mood2.get('arousal', 0) * (1-w)
            }

    # Handle cognitive context if present
    if 'cognitive' in state1 and 'cognitive' in state2:
        w = weights.get('cognitive', 0.1)
        
        # For raw state - merge message histories
        if is_raw:
            messages1 = state1['cognitive']
            messages2 = state2['cognitive']
            
            # Combine messages, keeping system messages and most relevant exchanges
            system_msgs = [m for m in messages1 + messages2 if m['role'] == 'system']
            other_msgs = [m for m in messages1 + messages2 if m['role'] != 'system']
            
            # Sort by timestamp if available, otherwise preserve order
            other_msgs.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Keep limited number of most recent exchanges
            blended['cognitive'] = system_msgs + other_msgs[:6]
            
        else:
            # For processed state - blend cognitive summaries if available
            cog1 = state1['cognitive']
            cog2 = state2['cognitive']
            
            if isinstance(cog1, dict) and isinstance(cog2, dict):
                blended['cognitive'] = {
                    k: cog1.get(k, '') if w >= 0.5 else cog2.get(k, '')
                    for k in set(cog1.keys()) | set(cog2.keys())
                }
            else:
                # If not dict format, keep the more recent one
                blended['cognitive'] = cog1 if w >= 0.5 else cog2

    return blended

def get_state_summary(state: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of a state dictionary.
    Useful for logging and debugging.
    
    Args:
        state: State dictionary to summarize
        
    Returns:
        str: Human-readable summary
    """
    summary = []
    
    if 'emotional_vectors' in state:
        emotions = state['emotional_vectors']
        strongest = max(emotions.items(), key=lambda x: x[1]) if emotions else None
        if strongest:
            summary.append(f"Emotion: {strongest[0]} ({strongest[1]:.2f})")
            
    if 'needs' in state:
        urgent_needs = sorted(
            [(need, data.get('urgency', 0)) 
             for need, data in state['needs'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:2]
        if urgent_needs:
            needs_str = ', '.join(f"{n} ({u:.2f})" for n, u in urgent_needs)
            summary.append(f"Urgent Needs: {needs_str}")
            
    if 'mood' in state:
        mood = state['mood']
        summary.append(
            f"Mood: valence={mood.get('valence', 0):.2f}, "
            f"arousal={mood.get('arousal', 0):.2f}"
        )
            
    return ' | '.join(summary)