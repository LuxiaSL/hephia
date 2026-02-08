"""
conflict.py

Contains conflict-related logic for cognitive merges.
Pulls in metrics from the orchestrator or prior detect_conflicts patterns.

We rely on:
 - detect_conflict(...) as a quick yes/no check
 - analyze_conflicts_for_synthesis(...) for deeper analysis
"""

from typing import Dict, Any, List, Optional, Union

import numpy as np
from ...metrics.orchestrator import RetrievalMetricsOrchestrator


def detect_cognitive_conflict(
    nodeA, nodeB,
    metrics: Dict[str, Any],
    metrics_orchestrator: Optional[RetrievalMetricsOrchestrator] = None
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
    if not metrics and metrics_orchestrator:
        from ...metrics.orchestrator import MetricsConfiguration
        metrics_config = MetricsConfiguration()
        metrics_config.detailed_metrics = True
        metrics = metrics_orchestrator.calculate_metrics(nodeA, nodeB)

    component_metrics = metrics.get('component_metrics', {})

    # If component_metrics contains raw float values (dissonance scores),
    # convert them to dict format expected by analyze_conflicts_for_synthesis
    if component_metrics and all(isinstance(v, (int, float, np.number)) for v in component_metrics.values()):
        mock_component_metrics = {}
        for comp_name, dissonance_score in component_metrics.items():
            if comp_name == 'semantic':
                mock_component_metrics['semantic'] = {
                    'embedding_similarity': 1.0 - float(dissonance_score),
                    'semantic_density': 0.5
                }
            elif comp_name == 'emotional':
                mock_component_metrics['emotional'] = {
                    'valence_shift': float(dissonance_score),
                    'intensity_delta': float(dissonance_score)
                }
            elif comp_name == 'temporal':
                mock_component_metrics['temporal'] = {
                    'temporal_drift': float(dissonance_score)
                }
            elif comp_name == 'strength':
                mock_component_metrics['strength'] = {
                    'strength_difference': float(dissonance_score)
                }

        analysis_metrics = {'component_metrics': mock_component_metrics}
    else:
        analysis_metrics = metrics

    analysis = analyze_conflicts_for_synthesis(nodeA, nodeB, analysis_metrics)

    if 'semantic' in analysis_metrics.get('component_metrics', {}):
        semantic = analysis_metrics['component_metrics']['semantic']
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

    semantic = cm.get('semantic', {})
    semantic_sim = semantic.get('embedding_similarity', 0.0)
    semantic_density = semantic.get('semantic_density', 0.0)

    emotional = cm.get('emotional', {})
    valence_shift = emotional.get('valence_shift', 0.0)
    intensity_delta = emotional.get('intensity_delta', 0.0)

    # Conflict conditions:
    # 1. High semantic similarity but emotional conflicts
    if semantic_sim > 0.8 and semantic_density > 0.6:
        if valence_shift > 0.6 or intensity_delta > 0.7:
            return True

    # 2. Overall conflict threshold
    conflict_score = (
        valence_shift * 0.5 +
        (semantic_sim * intensity_delta) * 0.5
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
    """
    has_conflicts = detect_conflict(child, parent, metrics)

    conflict_details = _analyze_complex_conflicts(child, parent, metrics)

    severity = conflict_details['severity'] if has_conflicts else 0.0
    requires_reflection = conflict_details['requires_reflection'] if has_conflicts else False

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
    Detailed analysis of potential memory conflicts for borderline merges:
    - Evaluate semantic drift, emotional inversions, temporal patterns
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
    divergences: List[Dict[str, Any]] = []

    # Semantic analysis
    semantic = cm.get('semantic', {})
    semantic_density = semantic.get('semantic_density', 0.0)
    embedding_sim = semantic.get('embedding_similarity', 0.0)

    if semantic_density > 0.7 and embedding_sim < 0.4:
        divergences.append({
            'type': 'semantic_drift',
            'severity': 1.0 - embedding_sim,
            'context': 'High density content with low embedding similarity'
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

    # Compute overall severity
    total_severity = sum(d['severity'] for d in divergences)
    severity = total_severity / max(len(divergences), 1) if divergences else 0.0

    requires_reflection = (severity > 0.6 or len(divergences) >= 3)

    return {
        'severity': severity,
        'requires_reflection': requires_reflection,
        'resolution_path': _suggest_resolution_path(divergences, severity),
        'key_divergences': divergences
    }


def _suggest_resolution_path(divergences: List[Dict[str, Any]], severity: float) -> str:
    """Suggest approach for resolving memory conflicts."""
    if not divergences:
        return "direct_merge"

    semantic_issues = sum(1 for d in divergences if 'semantic' in d['type'])
    emotional_issues = sum(1 for d in divergences if 'emotional' in d['type'])
    temporal_issues = sum(1 for d in divergences if 'temporal' in d['type'])

    if semantic_issues > emotional_issues and severity > 0.7:
        return "conscious_reflection"
    elif emotional_issues >= semantic_issues and severity > 0.6:
        return "emotional_processing"
    elif temporal_issues > 1:
        return "temporal_integration"
    elif severity > 0.8:
        return "deep_consolidation"
    else:
        return "gradual_integration"
