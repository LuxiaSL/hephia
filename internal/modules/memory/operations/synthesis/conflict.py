"""
conflict.py

Contains conflict-related logic for cognitive merges. 
Pulls in metrics from the orchestrator or prior detect_conflicts patterns.

We rely on:
 - detect_conflict(...) as a quick yes/no check
 - analyze_conflicts_for_synthesis(...) for deeper analysis
"""

from typing import Dict, Any, List

def detect_cognitive_conflict(
    nodeA, nodeB,
    metrics: Dict[str, Any],
    metrics_orchestrator=None
) -> Dict[str, Any]:
    """
    Single source of truth for cognitive memory conflict detection.
    
    Args:
        nodeA, nodeB: The nodes to check for conflicts
        metrics: Pre-calculated metrics if available
        metrics_orchestrator: Optional orchestrator to calculate metrics if not provided
        
    Returns:
        Dict containing:
        - has_conflicts: bool
        - severity: float
        - resolution_path: str
        - details: Dict with specific conflict info
    """
    # If metrics not provided, calculate them
    if not metrics and metrics_orchestrator:
        metrics = metrics_orchestrator.calculate_metrics(nodeA, nodeB)
    
    # Use existing analyze_conflicts_for_synthesis but with enhanced details
    analysis = analyze_conflicts_for_synthesis(nodeA, nodeB, metrics)
    
    # Add semantic analysis
    if 'semantic' in metrics.get('component_metrics', {}):
        semantic = metrics['component_metrics']['semantic']
        if semantic.get('semantic_density', 0) > 0.7:
            analysis['details']['semantic_context'] = {
                'density': semantic['semantic_density'],
                'embedding_similarity': semantic.get('embedding_similarity', 0)
            }
    
    return analysis

def detect_conflict(nodeA, nodeB, metrics: Dict[str, Any]) -> bool:
    """Quick check for conflicts between nodes using metrics."""
    if 'component_metrics' not in metrics:
        return False

    cm = metrics['component_metrics']
    
    # Get core metrics
    semantic = cm.get('semantic', {})
    semantic_sim = semantic.get('embedding_similarity', 0.0)
    semantic_density = semantic.get('semantic_density', 0.0)
    
    emotional = cm.get('emotional', {})
    valence_shift = emotional.get('valence_shift', 0.0)
    intensity_delta = emotional.get('intensity_delta', 0.0)
    
    state = cm.get('state', {})
    state_conflicts = _extract_state_conflicts(state)
    
    # Conflict conditions:
    # 1. High semantic similarity but emotional/state conflicts
    if semantic_sim > 0.8 and semantic_density > 0.6:
        if valence_shift > 0.6 or intensity_delta > 0.7:
            return True
        if state_conflicts > 0.7:
            return True
            
    # 2. Overall conflict threshold
    conflict_score = (
        state_conflicts * 0.4 +
        valence_shift * 0.3 +
        (semantic_sim * intensity_delta) * 0.3
    )
    
    return conflict_score > 0.75

def analyze_conflicts_for_synthesis(
    child: Any,
    parent: Any,
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyzes conflicts for a deeper synthesis decision:
    - Summarize overall conflict severity
    - Provide "resolution_path" suggestions (like direct_merge, reflection, etc.)
    - Indicate if we need additional strength for the new synthesis node

    Args:
        child, parent: CognitiveMemoryNode-like objects
        metrics: Pre-calculated retrieval metrics from orchestrator

    Returns:
        Dict with analysis:
         {
           'has_conflicts': bool,
           'severity': float,
           'requires_reflection': bool,
           'resolution_path': str,
           'additional_strength': float,
           'details': {...} # deeper conflict analysis
         }
    """
    # Basic yes/no conflict check
    has_conflicts = detect_conflict(child, parent, metrics)
    
    # Deeper analysis
    conflict_details = _analyze_complex_conflicts(child, parent, metrics)
    
    # If no conflicts found in basic check, severity might be small
    severity = conflict_details['severity'] if has_conflicts else 0.0
    requires_reflection = conflict_details['requires_reflection'] if has_conflicts else False

    # Additional strength for the new node can scale with conflict severity
    additional_strength = _calculate_synthesis_strength(severity, metrics.get('component_metrics', {}))

    return {
        'has_conflicts': has_conflicts,
        'severity': severity,
        'requires_reflection': requires_reflection,
        'resolution_path': conflict_details['resolution_path'],
        'additional_strength': additional_strength,
        'details': conflict_details
    }

def _calculate_synthesis_strength(conflict_severity: float, cm: Dict[str, Any]) -> float:
    """
    Helper that calculates how much 'extra' strength might be allocated 
    to the new node if we do a conflict-based synthesis.
    """
    base_strength = conflict_severity * 0.4
    emotional = cm.get('emotional', {})
    intensity = emotional.get('intensity_delta', 0.0)
    semantic = cm.get('semantic', {})
    semantic_density = semantic.get('semantic_density', 0.0)

    emotional_boost = intensity * 0.3
    semantic_boost = semantic_density * 0.2
    
    total = base_strength + emotional_boost + semantic_boost
    return min(1.0, total)

def _analyze_complex_conflicts(nodeA, nodeB, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    More detailed analysis of potential memory conflicts for borderline merges:
    - Evaluate semantic drift, emotional inversions, temporal patterns, state transitions
    - Summarize conflict severity & recommended resolution path
    """
    if 'component_metrics' not in metrics:
        return {
            'severity': 0.0,
            'requires_reflection': False,
            'resolution_path': 'direct_merge',
            'key_divergences': []
        }

    cm = metrics['component_metrics']
    divergences = []

    # Check semantic drift 
    semantic = cm.get('semantic', {})
    semantic_density = semantic.get('semantic_density', 0.0)
    semantic_sim = semantic.get('embedding_similarity', 0.0)
    
    # Only check cluster metrics if they were requested during calculation
    cluster_metrics = semantic.get('cluster_metrics', {})
    if cluster_metrics and semantic_density > 0.7:
        semantic_spread = cluster_metrics.get('semantic_spread', 0.0)
        if semantic_spread > 0.4:
            divergences.append({
                'type': 'semantic_drift',
                'severity': semantic_spread,
                'context': 'High density + spreading semantic field'
            })

    # Emotional mismatch
    emotional = cm.get('emotional', {})
    vector_sim = emotional.get('vector_similarity', 1.0)
    valence_shift = emotional.get('valence_shift', 0.0)
    intensity_delta = emotional.get('intensity_delta', 0.0)
    
    if vector_sim < 0.4 and valence_shift > 0.5:
        divergences.append({
            'type': 'emotional_inversion',
            'severity': valence_shift,
            'context': 'Opposing emotional valence in high-sim context'
        })
    
    if intensity_delta > 0.7:
        divergences.append({
            'type': 'emotional_intensity_shift',
            'severity': intensity_delta,
            'context': 'Large shift in emotional intensity'
        })

    # Temporal pattern analysis
    temporal = cm.get('temporal', {})
    # Only check patterns if analyze_temporal_patterns was called
    if 'patterns' in temporal:
        patterns = temporal.get('patterns', {})
        interval_consistency = patterns.get('interval_consistency', 1.0)
        recent_density = patterns.get('recent_density', 0.0)
        
        if interval_consistency < 0.4 and recent_density > 0.6:
            divergences.append({
                'type': 'temporal_pattern_break',
                'severity': 1 - interval_consistency,
                'context': 'Breaking established temporal patterns'
            })

    # State transitions and conflicts
    state = cm.get('state', {})
    # Check needs satisfaction and shifts
    needs = state.get('needs', {})
    satisfaction_sim = needs.get('satisfaction_similarity', 1.0)
    state_shifts = needs.get('state_shifts', 0.0)
    
    if satisfaction_sim < 0.3 or state_shifts > 0.7:
        divergences.append({
            'type': 'state_transition_stress',
            'severity': max(1 - satisfaction_sim, state_shifts),
            'context': 'Significant state transition detected'
        })
        
    # Check behavior transitions
    behavior = state.get('behavior', {})
    transition_sig = behavior.get('transition_significance', 0.0)
    if transition_sig > 0.7:
        divergences.append({
            'type': 'behavior_transition',
            'severity': transition_sig,
            'context': 'Major behavior pattern shift'
        })

    # Compute overall severity
    total_severity = sum(d['severity'] for d in divergences)
    severity = total_severity / max(len(divergences), 1) if divergences else 0.0

    # If severity is high or multiple divergences, reflection needed
    requires_reflection = (severity > 0.6 or len(divergences) >= 3)

    return {
        'severity': severity,
        'requires_reflection': requires_reflection,
        'resolution_path': _suggest_resolution_path(divergences, severity),
        'key_divergences': divergences
    }

def _suggest_resolution_path(divergences: List[Dict[str, Any]], severity: float) -> str:
    """
    Suggest approach for resolving memory conflicts.
    """
    if not divergences:
        return "direct_merge"

    # Tally up the conflict categories
    semantic_issues = sum(1 for d in divergences if 'semantic' in d['type'])
    emotional_issues = sum(1 for d in divergences if 'emotional' in d['type'])
    temporal_issues = sum(1 for d in divergences if 'temporal' in d['type'])
    state_issues = sum(1 for d in divergences if 'state' in d['type'])

    if semantic_issues > emotional_issues and severity > 0.7:
        return "conscious_reflection"
    elif emotional_issues >= semantic_issues and severity > 0.6:
        return "emotional_processing"
    elif temporal_issues > 1 or state_issues > 1:
        return "temporal_integration"
    elif severity > 0.8:
        return "deep_consolidation"
    else:
        return "gradual_integration"

def _extract_state_conflicts(state_metrics: Dict[str, Any]) -> float:
    """
    Interprets state sub-metrics to detect conflicts between nodes.
    Returns a float [0..1] indicating severity of state conflicts.
    """
    conflict_score = 0.0
    
    # Behavior mismatch
    behavior = state_metrics.get('behavior', {})
    if not behavior.get('matching', True):
        conflict_score += 0.3
    if behavior.get('transition_significance', 0) > 0.7:
        conflict_score += 0.2

    # Needs
    needs = state_metrics.get('needs', {})
    satisfaction_diff = 1.0 - needs.get('satisfaction_similarity', 1.0)
    if satisfaction_diff > 0.6:
        conflict_score += 0.25
    if needs.get('urgency_levels', 0) > 0.7:
        conflict_score += 0.25
    if needs.get('state_shifts', 0) > 0.5:
        conflict_score += 0.2

    # Mood
    mood = state_metrics.get('mood', {})
    if mood.get('similarity', 1.0) < 0.3:
        conflict_score += 0.2
    if mood.get('intensity_delta', 0) > 0.6:
        conflict_score += 0.2

    # Emotional
    emotional = state_metrics.get('emotional', {})
    if emotional.get('emotional_complexity', 1.0) < 0.3:
        conflict_score += 0.15
    if emotional.get('valence_shift', 0) > 0.6:
        conflict_score += 0.25
    if emotional.get('intensity_delta', 0) > 0.7:
        conflict_score += 0.2

    return min(1.0, conflict_score)
