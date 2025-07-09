#!/usr/bin/env python3
"""
Deep Semantic Significance Analysis Script
Parses detailed debug logs from significance calculations to identify
semantic discrimination bottlenecks and analyze novelty score distribution.
"""

import re
import json
import statistics
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple
import sys
from dataclasses import dataclass
import numpy as np

@dataclass
class ComponentAnalysis:
    """Analysis data for a single semantic component."""
    values: List[float]
    mean: float
    std_dev: float
    range_span: float
    coefficient_of_variation: float  # std_dev / mean
    
    @classmethod
    def from_values(cls, values: List[float]) -> 'ComponentAnalysis':
        if not values:
            return cls([], 0.0, 0.0, 0.0, 0.0)
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        range_val = max(values) - min(values)
        cv = std_val / mean_val if mean_val != 0 else 0.0
        
        return cls(values, mean_val, std_val, range_val, cv)

@dataclass 
class EvaluationEntry:
    """Single significance evaluation with semantic component details."""
    entry_id: str
    source_type: str
    
    # Final score
    final_significance: float
    
    # Component scores (novelty calculation)
    novelty_components: Dict[str, float]  # embedding_sim, text_rel, density
    novelty_final: float
    
    # Raw metrics before averaging
    raw_semantic_scores: List[Dict[str, float]]
    
    # Meta information
    comparison_count: int
    evaluation_method: str

class DeepSignificanceAnalyzer:
    """Deep analysis of semantic significance calculation logs."""
    
    def __init__(self):
        self.entries: List[EvaluationEntry] = []
        self.parsing_context = {}
        
        # Regex patterns for detailed parsing
        self.patterns = {
            # Entry identification
            'eval_start': re.compile(r'\[(?:SIG_|EVAL_)DEBUG\] Starting (?:significance|strength) evaluation'),
            'eval_source': re.compile(r'Memory Info: (?:Significance|Memory) evaluation complete for (\w+): ([0-9.]+)'),
            
            # Component analysis  
            'novelty_start': re.compile(r'\[NOVELTY_DEBUG\] Processing (\d+) metric entries'),
            'novelty_entry': re.compile(r'\[NOVELTY_DEBUG\] Entry (\d+): emb_sim=([0-9.]+), text_rel=([0-9.]+), density=([0-9.]+), novelty=([0-9.]+)'),
            'novelty_final': re.compile(r'\[NOVELTY_DEBUG\] Final novelty: ([0-9.]+) from (\d+) scores'),

            # Final evaluation
            'eval_complete': re.compile(r'\[(?:SIG_|EVAL_)DEBUG\] (?:Significance|Memory) evaluation complete: novelty=([0-9.]+)'),
            'eval_method': re.compile(r'method: ([^,\)]+)'),
            'comparison_count': re.compile(r'comparisons: (\d+)')
        }
        
    def parse_log_file(self, filepath: str) -> None:
        """Parse log file and extract detailed semantic component analysis."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        current_entry = {}
        in_evaluation = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check for evaluation start
            if self.patterns['eval_start'].search(line):
                if current_entry:  # Save previous entry
                    self._finalize_entry(current_entry)
                current_entry = self._init_entry(i)
                in_evaluation = True
                continue
                
            if not in_evaluation:
                continue
                
            # Parse source type and final score
            source_match = self.patterns['eval_source'].search(line)
            if source_match:
                current_entry['source_type'] = source_match.group(1)
                # This is the overall score, might not be the semantic-only score
                current_entry['overall_score'] = float(source_match.group(2))
                continue
                
            # Parse component analysis
            self._parse_novelty_analysis(line, current_entry)
            
            # Parse final evaluation score (from novelty)
            eval_complete = self.patterns['eval_complete'].search(line)
            if eval_complete:
                current_entry['novelty_final'] = float(eval_complete.group(1))
                
            # Parse meta info
            method_match = self.patterns['eval_method'].search(line)
            if method_match:
                current_entry['evaluation_method'] = method_match.group(1)
                
            count_match = self.patterns['comparison_count'].search(line)
            if count_match:
                current_entry['comparison_count'] = int(count_match.group(1))
                
        # Finalize last entry
        if current_entry:
            self._finalize_entry(current_entry)
            
    def _init_entry(self, line_num: int) -> Dict[str, Any]:
        """Initialize a new entry."""
        return {
            'entry_id': f'eval_{line_num}',
            'source_type': 'unknown',
            'novelty_components': {},
            'raw_semantic_scores': [],
            'comparison_count': 0,
            'evaluation_method': 'unknown'
        }
        
    def _parse_novelty_analysis(self, line: str, entry: Dict[str, Any]) -> None:
        """Parse novelty component analysis."""
        entry_match = self.patterns['novelty_entry'].search(line)
        if entry_match:
            semantic_data = {
                'embedding_similarity': float(entry_match.group(2)),
                'text_relevance': float(entry_match.group(3)), 
                'semantic_density': float(entry_match.group(4)),
                'novelty_score': float(entry_match.group(5))
            }
            entry['raw_semantic_scores'].append(semantic_data)
            
        final_match = self.patterns['novelty_final'].search(line)
        if final_match:
            entry['novelty_final'] = float(final_match.group(1))
            entry['novelty_comparison_count'] = int(final_match.group(2))
            
    def _finalize_entry(self, entry_data: Dict[str, Any]) -> None:
        """Convert parsed data to EvaluationEntry and store."""
        try:
            # Extract component summaries
            novelty_components = {}
            if entry_data['raw_semantic_scores']:
                semantic_scores = entry_data['raw_semantic_scores']
                novelty_components = {
                    'embedding_similarity': [s['embedding_similarity'] for s in semantic_scores],
                    'text_relevance': [s['text_relevance'] for s in semantic_scores],  
                    'semantic_density': [s['semantic_density'] for s in semantic_scores],
                    'novelty_score': [s['novelty_score'] for s in semantic_scores]
                }
                
            novelty_final_score = entry_data.get('novelty_final', 0.0)
                
            eval_entry = EvaluationEntry(
                entry_id=entry_data['entry_id'],
                source_type=entry_data['source_type'],
                final_significance=novelty_final_score,
                novelty_components=novelty_components,
                novelty_final=novelty_final_score,
                raw_semantic_scores=entry_data['raw_semantic_scores'],
                comparison_count=entry_data.get('comparison_count', 0),
                evaluation_method=entry_data.get('evaluation_method', 'unknown')
            )
            
            self.entries.append(eval_entry)
            
        except Exception as e:
            print(f"Warning: Failed to finalize entry {entry_data.get('entry_id', 'unknown')}: {e}")
            
    def analyze_discrimination_bottlenecks(self) -> Dict[str, Any]:
        """Analyze where discrimination is being lost in the semantic pipeline."""
        if not self.entries:
            return {"error": "No entries found"}
            
        analysis = {
            'component_discrimination': {},
            'averaging_impact': {},
            'bottleneck_identification': {},
            'reward_function_analysis': {}
        }
        
        # 1. Component-level discrimination analysis
        analysis['component_discrimination'] = self._analyze_component_discrimination()
        
        # 2. Impact of averaging on discrimination
        analysis['averaging_impact'] = self._analyze_averaging_impact()
        
        # 3. Identify specific bottlenecks
        analysis['bottleneck_identification'] = self._identify_bottlenecks()
        
        # 4. Reward function characteristics (based on final novelty score)
        analysis['reward_function_analysis'] = self._analyze_reward_function()
        
        return analysis
        
    def _analyze_component_discrimination(self) -> Dict[str, Any]:
        """Analyze discrimination power of each semantic component."""
        component_analysis = {}
        
        # Semantic/novelty components
        if any(entry.novelty_components for entry in self.entries):
            # Aggregate all semantic component values
            all_embedding_sim = []
            all_text_rel = []
            all_semantic_density = []
            all_novelty = []
            
            for entry in self.entries:
                if entry.novelty_components:
                    all_embedding_sim.extend(entry.novelty_components.get('embedding_similarity', []))
                    all_text_rel.extend(entry.novelty_components.get('text_relevance', []))
                    all_semantic_density.extend(entry.novelty_components.get('semantic_density', []))
                    all_novelty.extend(entry.novelty_components.get('novelty_score', []))
                    
            semantic_data = {
                'embedding_similarity': ComponentAnalysis.from_values(all_embedding_sim),
                'text_relevance': ComponentAnalysis.from_values(all_text_rel),
                'semantic_density': ComponentAnalysis.from_values(all_semantic_density),
                'novelty_score': ComponentAnalysis.from_values(all_novelty)
            }
            
            component_analysis['semantic'] = semantic_data
            
        return component_analysis
        
    def _analyze_averaging_impact(self) -> Dict[str, Any]:
        """Analyze how averaging semantic scores reduces discrimination."""
        averaging_analysis = {}
        
        for entry in self.entries:
            entry_analysis = {}
            
            # Semantic averaging impact
            if entry.raw_semantic_scores:
                raw_novelties = [s['novelty_score'] for s in entry.raw_semantic_scores]
                if len(raw_novelties) > 1:
                    raw_std = statistics.stdev(raw_novelties)
                    raw_range = max(raw_novelties) - min(raw_novelties)
                    final_novelty = entry.novelty_final
                    
                    entry_analysis['semantic'] = {
                        'raw_std': raw_std,
                        'raw_range': raw_range,
                        'raw_values': raw_novelties,
                        'final_value': final_novelty,
                        'discrimination_loss': raw_std  # Higher std = more lost discrimination
                    }
                    
            if entry_analysis:
                averaging_analysis[entry.entry_id] = entry_analysis
                
        return averaging_analysis
        
    def _identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify specific bottlenecks in the discrimination pipeline.""" 
        bottlenecks = {}
        
        # Check component-level bottlenecks
        component_disc = self._analyze_component_discrimination()
        
        for component_type, components in component_disc.items():
            bottlenecks[f'{component_type}_bottlenecks'] = {}
            
            for comp_name, analysis in components.items():
                if hasattr(analysis, 'coefficient_of_variation'):
                    cv = analysis.coefficient_of_variation
                    range_span = analysis.range_span
                    
                    bottleneck_score = 0.0
                    issues = []
                    
                    if cv < 0.15:  # Low variance relative to mean
                        bottleneck_score += 0.4
                        issues.append("low_variance")
                        
                    if range_span < 0.2:  # Narrow range
                        bottleneck_score += 0.3
                        issues.append("narrow_range")
                        
                    if analysis.std_dev < 0.05:  # Very low standard deviation
                        bottleneck_score += 0.3
                        issues.append("minimal_spread")
                        
                    bottlenecks[f'{component_type}_bottlenecks'][comp_name] = {
                        'bottleneck_score': bottleneck_score,
                        'issues': issues,
                        'coefficient_of_variation': cv,
                        'range_span': range_span,
                        'std_dev': analysis.std_dev
                    }
                    
        return bottlenecks
        
    def _analyze_reward_function(self) -> Dict[str, Any]:
        """Analyze final novelty score characteristics like an RL reward function."""
        reward_analysis = {}
        
        final_scores = [entry.final_significance for entry in self.entries if entry.final_significance > 0]
        
        if not final_scores:
            return {"error": "No final scores available"}
            
        # Reward distribution analysis
        reward_analysis['distribution'] = {
            'mean': statistics.mean(final_scores),
            'std_dev': statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0,
            'skewness': self._calculate_skewness(final_scores),
            'kurtosis': self._calculate_kurtosis(final_scores),
            'range_span': max(final_scores) - min(final_scores),
            'effective_range': self._calculate_effective_range(final_scores)
        }
        
        # Reward signal quality
        reward_analysis['signal_quality'] = {
            'signal_to_noise_ratio': self._calculate_snr(final_scores),
            'discrimination_index': self._calculate_discrimination_index(final_scores),
            'reward_sparsity': self._calculate_reward_sparsity(final_scores),
            'gradient_availability': self._calculate_gradient_availability(final_scores)
        }
        
        # Actionability analysis
        reward_analysis['actionability'] = {
            'threshold_sensitivity': self._analyze_threshold_sensitivity(final_scores),
            'optimization_landscape': self._analyze_optimization_landscape(final_scores)
        }
        
        return reward_analysis
        
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of distribution."""
        if len(values) < 3:
            return 0.0
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        if std_val == 0:
            return 0.0
        skewness = sum(((x - mean_val) / std_val) ** 3 for x in values) / len(values)
        return skewness
        
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of distribution."""
        if len(values) < 4:
            return 0.0
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        if std_val == 0:
            return 0.0
        kurtosis = sum(((x - mean_val) / std_val) ** 4 for x in values) / len(values) - 3
        return kurtosis
        
    def _calculate_effective_range(self, values: List[float]) -> float:
        """Calculate effective range (90th percentile range).""" 
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        p5_idx = int(0.05 * n)
        p95_idx = int(0.95 * n) -1
        return sorted_vals[p95_idx] - sorted_vals[p5_idx]
        
    def _calculate_snr(self, values: List[float]) -> float:
        """Calculate signal-to-noise ratio."""
        if len(values) < 2:
            return 0.0
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        return mean_val / std_val if std_val != 0 else float('inf')
        
    def _calculate_discrimination_index(self, values: List[float]) -> float:
        """Calculate discrimination index (coefficient of variation)."""
        if len(values) < 2:
            return 0.0
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        return std_val / mean_val if mean_val != 0 else 0.0
        
    def _calculate_reward_sparsity(self, values: List[float]) -> float:
        """Calculate how sparse/dense the rewards are."""
        if not values:
            return 1.0
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        if std_val == 0:
            return 1.0  # Maximum sparsity
        meaningful_threshold = std_val * 0.5
        meaningful_count = sum(1 for v in values if abs(v - mean_val) > meaningful_threshold)
        return 1.0 - (meaningful_count / len(values))
        
    def _calculate_gradient_availability(self, values: List[float]) -> float:
        """Calculate how much gradient information is available."""
        if len(values) < 2:
            return 0.0
        range_span = max(values) - min(values)
        relative_range = range_span / max(values) if max(values) != 0 else 0.0
        return relative_range
        
    def _analyze_threshold_sensitivity(self, values: List[float]) -> Dict[str, Any]:
        """Analyze how sensitive the system is to threshold changes."""
        thresholds = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        sensitivity_analysis = {}
        prev_acceptance = None
        
        for threshold in thresholds:
            acceptance_rate = sum(1 for v in values if v >= threshold) / len(values)
            sensitivity_analysis[f'threshold_{threshold}'] = {
                'acceptance_rate': acceptance_rate,
                'accepted_count': sum(1 for v in values if v >= threshold)
            }
            if prev_acceptance is not None:
                change = abs(acceptance_rate - prev_acceptance)
                sensitivity_analysis[f'threshold_{threshold}']['change_from_previous'] = change
            prev_acceptance = acceptance_rate
        return sensitivity_analysis
        
    def _analyze_optimization_landscape(self, values: List[float]) -> Dict[str, Any]:
        """Analyze the optimization landscape characteristics."""
        landscape = {}
        if not values:
            return {'flatness': 1.0, 'smoothness': 1.0}
            
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total_range = max(sorted_vals) - min(sorted_vals)
        
        if total_range == 0:
            landscape['flatness'] = 1.0
        else:
            tight_band_width = total_range * 0.1
            tight_band_count = 0
            for i in range(n - 1):
                if sorted_vals[i + 1] - sorted_vals[i] <= tight_band_width:
                    tight_band_count += 1
            landscape['flatness'] = tight_band_count / (n - 1) if n > 1 else 1.0
            
        if n > 2:
            differences = [sorted_vals[i + 1] - sorted_vals[i] for i in range(n - 1)]
            mean_diff = statistics.mean(differences)
            stdev_diff = statistics.stdev(differences)
            landscape['smoothness'] = 1.0 - (stdev_diff / mean_diff) if mean_diff != 0 else 1.0
        else:
            landscape['smoothness'] = 1.0
            
        return landscape
        
    def print_deep_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print comprehensive deep analysis report."""
        print("=" * 80)
        print("DEEP SEMANTIC SIGNIFICANCE DISCRIMINATION ANALYSIS")
        print("=" * 80)
        
        print(f"\nFound {len(self.entries)} detailed evaluation entries")
        
        # Component discrimination analysis
        if 'component_discrimination' in analysis and analysis['component_discrimination']:
            print(f"\n{'='*60}")
            print("SEMANTIC COMPONENT DISCRIMINATION ANALYSIS")
            print(f"{'='*60}")
            
            comp_disc = analysis['component_discrimination']
            
            for component_type, components in comp_disc.items():
                print(f"\n{component_type.upper()} COMPONENTS:")
                print("-" * 40)
                
                for comp_name, comp_analysis in components.items():
                    if hasattr(comp_analysis, 'coefficient_of_variation'):
                        print(f"{comp_name:25s}: mean={comp_analysis.mean:.3f}, "
                              f"std={comp_analysis.std_dev:.3f}, "
                              f"range={comp_analysis.range_span:.3f}, "
                              f"cv={comp_analysis.coefficient_of_variation:.3f}")
                        
                        issues = []
                        if comp_analysis.coefficient_of_variation < 0.15:
                            issues.append("LOW_VARIANCE")
                        if comp_analysis.range_span < 0.2:
                            issues.append("NARROW_RANGE")
                        if comp_analysis.std_dev < 0.05:
                            issues.append("MINIMAL_SPREAD")
                            
                        if issues:
                            print(f"{'':25s}  ⚠️  DISCRIMINATION ISSUES: {', '.join(issues)}")
                            
        # Bottleneck identification  
        if 'bottleneck_identification' in analysis and analysis['bottleneck_identification']:
            print(f"\n{'='*60}")
            print("DISCRIMINATION BOTTLENECK IDENTIFICATION")
            print(f"{'='*60}")
            
            bottlenecks = analysis['bottleneck_identification']
            
            for component_type, component_bottlenecks in bottlenecks.items():
                if component_bottlenecks:
                    print(f"\n{component_type.upper().replace('_BOTTLENECKS', '')} BOTTLENECKS:")
                    print("-" * 40)
                    
                    for comp_name, bottleneck_info in component_bottlenecks.items():
                        score = bottleneck_info['bottleneck_score']
                        if score > 0:
                            severity = "🔴 CRITICAL" if score >= 0.7 else "🟡 MODERATE" if score >= 0.4 else "🟢 MINOR"
                            print(f"{comp_name:25s}: {severity} (score: {score:.2f})")
                            print(f"{'':25s}  Issues: {', '.join(bottleneck_info['issues'])}")
                            print(f"{'':25s}  CV: {bottleneck_info['coefficient_of_variation']:.3f}, "
                                  f"Range: {bottleneck_info['range_span']:.3f}, "
                                  f"StdDev: {bottleneck_info['std_dev']:.3f}")
                        
        # Reward function analysis
        if 'reward_function_analysis' in analysis and 'distribution' in analysis['reward_function_analysis']:
            print(f"\n{'='*60}")
            print("NOVELTY SCORE ANALYSIS (RL Perspective)")
            print(f"{'='*60}")
            
            reward_analysis = analysis['reward_function_analysis']
            
            if 'distribution' in reward_analysis:
                dist = reward_analysis['distribution']
                print(f"\nScore Distribution:")
                print(f"  Mean:                  {dist.get('mean', 0):.3f}")
                print(f"  Std Dev:               {dist.get('std_dev', 0):.3f}")
                print(f"  Skewness:              {dist.get('skewness', 0):.3f}")
                print(f"  Kurtosis:              {dist.get('kurtosis', 0):.3f}")
                print(f"  Range:                 {dist.get('range_span', 0):.3f}")
                print(f"  Effective Range (90%): {dist.get('effective_range', 0):.3f}")
                
            if 'signal_quality' in reward_analysis:
                signal = reward_analysis['signal_quality']
                print(f"\nSignal Quality:")
                print(f"  Signal-to-Noise:       {signal.get('signal_to_noise_ratio', 0):.3f}")
                print(f"  Discrimination Index:  {signal.get('discrimination_index', 0):.3f}")
                print(f"  Reward Sparsity:       {signal.get('reward_sparsity', 0):.3f}")
                print(f"  Gradient Availability: {signal.get('gradient_availability', 0):.3f}")
                
                disc_idx = signal.get('discrimination_index', 0)
                sparsity = signal.get('reward_sparsity', 0)
                
                print(f"\n  Interpretation:")
                if disc_idx < 0.1:
                    print(f"    🔴 Very poor discrimination (DI: {disc_idx:.3f})")
                elif disc_idx < 0.2:
                    print(f"    🟡 Poor discrimination (DI: {disc_idx:.3f})")
                else:
                    print(f"    🟢 Good discrimination (DI: {disc_idx:.3f})")
                    
                if sparsity > 0.8:
                    print(f"    🔴 Very sparse rewards (SP: {sparsity:.3f})")
                elif sparsity > 0.6:
                    print(f"    🟡 Sparse rewards (SP: {sparsity:.3f})")
                else:
                    print(f"    🟢 Dense rewards (SP: {sparsity:.3f})")
                    
            if 'actionability' in reward_analysis and 'threshold_sensitivity' in reward_analysis['actionability']:
                thresh_sens = reward_analysis['actionability']['threshold_sensitivity']
                print(f"\nThreshold Sensitivity:")
                
                for threshold_key, threshold_data in thresh_sens.items():
                    threshold = threshold_key.split('_')[1]
                    acceptance = threshold_data['acceptance_rate']
                    change = threshold_data.get('change_from_previous', 0)
                    
                    change_desc = f"(Δ{change:.3f})" if 'change_from_previous' in threshold_data else ""
                    print(f"  Threshold @ {threshold}: {acceptance:.1%} accepted {change_desc}")
                    
        # Recommendations
        print(f"\n{'='*60}")
        print("DISCRIMINATION ENHANCEMENT RECOMMENDATIONS")
        print(f"{'='*60}")
        
        self._generate_recommendations(analysis)
        
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> None:
        """Generate specific recommendations for improving semantic discrimination."""
        recommendations = []
        
        # Check component discrimination
        if 'component_discrimination' in analysis and 'semantic' in analysis['component_discrimination']:
            semantic = analysis['component_discrimination']['semantic']
            
            if hasattr(semantic.get('semantic_density'), 'coefficient_of_variation'):
                density_cv = semantic['semantic_density'].coefficient_of_variation
                if density_cv < 0.15:
                    recommendations.append(
                        f"🔧 SEMANTIC DENSITY: The 'semantic_density' component has a low coefficient of variation ({density_cv:.3f}). "
                        "Consider applying a more aggressive sigmoid or non-linear transformation to spread values out."
                    )
                    
            if hasattr(semantic.get('embedding_similarity'), 'range_span'):
                emb_range = semantic['embedding_similarity'].range_span
                if emb_range < 0.3:
                    recommendations.append(
                        f"🔧 EMBEDDING SIMILARITY: The value range is very narrow ({emb_range:.3f}). "
                        "Investigate using different dimensionality reduction techniques or alternative similarity metrics (e.g., Manhattan distance) to improve separation."
                    )
                        
        # Check reward function quality
        if 'reward_function_analysis' in analysis and 'signal_quality' in analysis['reward_function_analysis']:
            signal = analysis['reward_function_analysis']['signal_quality']
            
            disc_idx = signal.get('discrimination_index', 0)
            if disc_idx < 0.15:
                recommendations.append(
                    "🚨 CRITICAL: The final novelty score has a very poor discrimination index. "
                    "This indicates that most evaluated items receive a very similar score, making it hard to distinguish between them. Focus on improving the underlying component metrics."
                )
                
        # Check bottlenecks
        if 'bottleneck_identification' in analysis:
            critical_bottlenecks = []
            for comp_name, bottleneck_info in analysis['bottleneck_identification'].get('semantic_bottlenecks', {}).items():
                if bottleneck_info['bottleneck_score'] >= 0.7:
                    critical_bottlenecks.append(comp_name)
                    
            if critical_bottlenecks:
                recommendations.append(
                    f"🔴 CRITICAL BOTTLENECKS: The following components are severely limiting discrimination: {', '.join(critical_bottlenecks)}. "
                    "These components need significant redesign or tuning."
                )
                
        # Check averaging impact
        if 'averaging_impact' in analysis:
            avg_impact = analysis['averaging_impact']
            high_loss_entries = [
                entry_id for entry_id, entry_data in avg_impact.items()
                if entry_data.get('semantic', {}).get('discrimination_loss', 0) > 0.15
            ]
            
            if len(high_loss_entries) > len(avg_impact) * 0.3:  # More than 30% have high loss
                recommendations.append(
                    "📊 AVERAGING IMPACT: Simple averaging appears to be washing out significant variations in novelty scores across comparisons. "
                    "Consider alternative aggregation methods like max(), min(), or a weighted average that prioritizes more discriminative comparisons."
                )
                
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec}")
        else:
            print("\n✅ No critical semantic discrimination issues identified - system appears well-tuned.")
            
        # Always show potential enhancements
        print(f"\n🚀 POTENTIAL ENHANCEMENTS:")
        print(f"1. Add semantic territory tracking for adaptive novelty thresholds.")
        print(f"2. Consider ensemble methods for component aggregation (e.g., geometric mean).")
        print(f"3. Implement novelty decay based on temporal or access patterns.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python deep_significance_analyzer.py <log_file_path>")
        print("\nThis script performs deep analysis of SEMANTIC significance calculation logs")
        print("to identify discrimination bottlenecks and analyze novelty score distribution.")
        sys.exit(1)
        
    log_file = sys.argv[1]
    
    analyzer = DeepSignificanceAnalyzer()
    
    try:
        print("Parsing detailed semantic significance logs...")
        analyzer.parse_log_file(log_file)
        
        print(f"Analyzing discrimination patterns...")
        analysis = analyzer.analyze_discrimination_bottlenecks()
        
        analyzer.print_deep_analysis(analysis)
        
    except FileNotFoundError:
        print(f"Error: Could not find log file '{log_file}'")
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()