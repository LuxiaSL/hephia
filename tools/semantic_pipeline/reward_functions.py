"""
reward_functions.py

Neural optimization reward functions for semantic memory discrimination.
Implements the 4-component reward system designed for Bayesian optimization
of neuromorphic memory patterns.

Based on reward_function_design_philosophy.md requirements:
- CV sweet spot targeting (0.4-0.8)
- Balance over single-metric dominance  
- Smooth, stable discrimination patterns
- Anti-gaming measures for extreme parameters
"""

import statistics
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class RewardAnalysis:
    """Detailed breakdown of reward calculation for interpretability."""
    discrimination_score: float
    balance_score: float
    robustness_score: float
    stability_score: float
    composite_reward: float
    
    # Detailed breakdowns
    component_cvs: Dict[str, float]
    cv_penalties: Dict[str, float]
    balance_variance: float
    coverage_ratio: float
    parameter_penalties: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/analysis."""
        return {
            'scores': {
                'discrimination': self.discrimination_score,
                'balance': self.balance_score,
                'robustness': self.robustness_score,
                'stability': self.stability_score,
                'composite': self.composite_reward
            },
            'details': {
                'component_cvs': self.component_cvs,
                'cv_penalties': self.cv_penalties,
                'balance_variance': self.balance_variance,
                'coverage_ratio': self.coverage_ratio,
                'parameter_penalties': self.parameter_penalties
            }
        }


class DiscriminationRewardFunction:
    """
    Core reward function for neural optimization of semantic discrimination.
    
    Implements neuromorphic memory principles:
    - Balanced multi-dimensional cognitive coverage
    - Smooth discrimination patterns over jagged ones
    - Anti-gaming measures for parameter stability
    - Robustness across different databases/providers
    """
    
    def __init__(self, 
                 discrimination_weight: float = 0.40,
                 balance_weight: float = 0.25,
                 robustness_weight: float = 0.20,
                 stability_weight: float = 0.15):
        """
        Initialize reward function with component weights.
        
        Args:
            discrimination_weight: Weight for CV discrimination score (40%)
            balance_weight: Weight for balance/coverage score (25%) 
            robustness_weight: Weight for cross-database robustness (20%)
            stability_weight: Weight for parameter stability (15%)
        """
        self.discrimination_weight = discrimination_weight
        self.balance_weight = balance_weight
        self.robustness_weight = robustness_weight
        self.stability_weight = stability_weight
        
        # Validate weights sum to 1.0
        total_weight = sum([discrimination_weight, balance_weight, robustness_weight, stability_weight])
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Reward weights must sum to 1.0, got {total_weight:.3f}")
    
    def calculate_discrimination_reward(self, component_cvs: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate discrimination reward based on CV sweet spot targeting.
        
        Rewards components in 0.4-0.8 range, penalizes extremes.
        
        Args:
            component_cvs: Dictionary mapping component names to CV values
            
        Returns:
            (discrimination_score, cv_penalties_dict)
        """
        cv_rewards = {}
        
        for comp_name, cv in component_cvs.items():
            if 0.4 <= cv <= 0.8:
                # Optimal range - full reward with bonus for center
                center_bonus = 1.0 - abs(cv - 0.6) / 0.2 * 0.1  # Small bonus for 0.6 center
                reward = min(1.0, center_bonus)
            elif 0.3 <= cv < 0.4 or 0.8 < cv <= 1.0:
                # Acceptable but suboptimal - partial reward
                if cv < 0.4:
                    reward = 0.6 + (cv - 0.3) / 0.1 * 0.3  # 0.6-0.9 linear
                else:
                    reward = 0.9 - (cv - 0.8) / 0.2 * 0.3  # 0.9-0.6 linear
            elif 0.2 <= cv < 0.3 or 1.0 < cv <= 1.2:
                # Poor but not broken - minimal reward
                if cv < 0.3:
                    reward = 0.2 + (cv - 0.2) / 0.1 * 0.4  # 0.2-0.6 linear
                else:
                    reward = 0.6 - (cv - 1.0) / 0.2 * 0.4  # 0.6-0.2 linear
            else:
                # Too extreme - penalty
                reward = 0.0
                
            cv_rewards[comp_name] = reward
        
        discrimination_score = statistics.mean(cv_rewards.values()) if cv_rewards else 0.0
        return discrimination_score, cv_rewards
    
    def calculate_balance_reward(self, component_cvs: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Calculate balance reward to prevent single-metric dominance.
        
        Penalizes high variance across components, rewards broad coverage.
        
        Args:
            component_cvs: Dictionary mapping component names to CV values
            
        Returns:
            (balance_score, cv_variance, coverage_ratio)
        """
        if not component_cvs:
            return 0.0, 0.0, 0.0
            
        cv_values = list(component_cvs.values())
        
        # Balance metric: penalize high variance across components
        cv_variance = statistics.variance(cv_values) if len(cv_values) > 1 else 0.0
        balance_score = max(0.0, 1.0 - (cv_variance * 2.0))  # Penalize jaggedness
        
        # Coverage metric: reward having most components above threshold
        above_threshold = sum(1 for cv in cv_values if cv >= 0.3)
        coverage_ratio = above_threshold / len(cv_values)
        
        # Combine balance and coverage (60/40 split)
        final_balance_score = 0.6 * balance_score + 0.4 * coverage_ratio
        
        return final_balance_score, cv_variance, coverage_ratio
    
    def calculate_robustness_reward(self, cross_db_results: List[Dict[str, Any]]) -> float:
        """
        Calculate robustness reward for consistency across databases/providers.
        
        Args:
            cross_db_results: List of experiment results from different databases
            
        Returns:
            robustness_score (0.0-1.0)
        """
        if len(cross_db_results) < 2:
            return 1.0  # Single database optimization - no penalty
        
        # Extract discrimination scores from each database
        db_discrimination_scores = []
        
        for db_result in cross_db_results:
            # Extract component CVs from this database result
            component_cvs = self._extract_component_cvs_from_result(db_result)
            if component_cvs:
                discrimination_score, _ = self.calculate_discrimination_reward(component_cvs)
                db_discrimination_scores.append(discrimination_score)
        
        if len(db_discrimination_scores) < 2:
            return 1.0  # Cannot calculate variance
        
        # Low variance across databases = high robustness
        variance = statistics.variance(db_discrimination_scores)
        robustness_score = max(0.0, 1.0 - (variance * 4.0))
        
        return robustness_score
    
    def calculate_stability_penalty(self, 
                                   parameter_vector: np.ndarray, 
                                   parameter_bounds: Dict[str, Tuple[float, float]]) -> Tuple[float, List[float]]:
        """
        Calculate stability penalty for extreme parameter values.
        
        Penalizes parameters too close to bounds (gaming prevention).
        
        Args:
            parameter_vector: Parameter values from optimization
            parameter_bounds: Dictionary mapping parameter names to (min, max) bounds
            
        Returns:
            (stability_score, penalty_list)
        """
        if len(parameter_vector) != len(parameter_bounds):
            # Fallback: assume uniform penalty if length mismatch
            return 0.8, [0.8] * len(parameter_vector)
        
        penalties = []
        bounds_values = list(parameter_bounds.values())
        
        for param_val, (min_bound, max_bound) in zip(parameter_vector, bounds_values):
            param_range = max_bound - min_bound
            if param_range <= 0:
                penalties.append(1.0)  # No penalty for invalid bounds
                continue
                
            normalized_val = (param_val - min_bound) / param_range
            
            # Penalize being too close to bounds (within 5%)
            if normalized_val < 0.05 or normalized_val > 0.95:
                penalty = 0.5  # Moderate penalty
            elif normalized_val < 0.1 or normalized_val > 0.9:
                penalty = 0.8  # Light penalty
            else:
                penalty = 1.0  # No penalty
                
            penalties.append(penalty)
        
        stability_score = statistics.mean(penalties) if penalties else 1.0
        return stability_score, penalties
    
    def composite_reward(self, 
                        experiment_results: Dict[str, Any],
                        parameter_vector: np.ndarray,
                        parameter_bounds: Dict[str, Tuple[float, float]],
                        cross_db_results: Optional[List[Dict[str, Any]]] = None) -> RewardAnalysis:
        """
        Calculate composite reward function for Bayesian optimization.
        
        Args:
            experiment_results: Results from running semantic test harness
            parameter_vector: Parameter vector being evaluated
            parameter_bounds: Parameter bounds for stability checking
            cross_db_results: Optional cross-database results for robustness
            
        Returns:
            RewardAnalysis with detailed breakdown
        """
        # Extract component CVs from experiment results
        component_cvs = self._extract_component_cvs_from_result(experiment_results)
        
        # Calculate component scores
        discrimination_score, cv_penalties = self.calculate_discrimination_reward(component_cvs)
        balance_score, balance_variance, coverage_ratio = self.calculate_balance_reward(component_cvs)
        robustness_score = self.calculate_robustness_reward(cross_db_results or [])
        stability_score, parameter_penalties = self.calculate_stability_penalty(parameter_vector, parameter_bounds)
        
        # Weighted combination emphasizing discrimination but ensuring balance
        composite = (
            self.discrimination_weight * discrimination_score +
            self.balance_weight * balance_score +
            self.robustness_weight * robustness_score +
            self.stability_weight * stability_score
        )
        
        return RewardAnalysis(
            discrimination_score=discrimination_score,
            balance_score=balance_score,
            robustness_score=robustness_score,
            stability_score=stability_score,
            composite_reward=composite,
            component_cvs=component_cvs,
            cv_penalties=cv_penalties,
            balance_variance=balance_variance,
            coverage_ratio=coverage_ratio,
            parameter_penalties=parameter_penalties
        )
    
    def _extract_component_cvs_from_result(self, experiment_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract component CV values from experiment result structure.
        Updated for 5-component system - topic_surprise removed.
        
        Args:
            experiment_result: Result dictionary from semantic test harness
            
        Returns:
            Dictionary mapping component names to CV values
        """
        component_cvs = {}
        
        # Try to extract from analysis_summary.component_discrimination_analysis
        analysis_summary = experiment_result.get('analysis_summary', {})
        component_analysis = analysis_summary.get('component_discrimination_analysis', {})
        
        # Expected 5 components (topic_surprise removed)
        expected_components = {
            'ne_density', 'conceptual_surprise', 'logical_complexity', 
            'conceptual_bridging', 'information_density'
        }
        
        for comp_name, metrics in component_analysis.items():
            # Only process expected components (filter out topic_surprise if it somehow appears)
            if comp_name in expected_components and isinstance(metrics, dict) and 'coefficient_of_variation' in metrics:
                component_cvs[comp_name] = metrics['coefficient_of_variation']
        
        # Fallback: try direct CV extraction from other locations
        if not component_cvs:
            # Try semantic_density_analysis for overall CV
            density_analysis = analysis_summary.get('semantic_density_analysis', {})
            if 'distribution' in density_analysis:
                dist = density_analysis['distribution']
                if 'coefficient_of_variation' in dist:
                    component_cvs['semantic_density'] = dist['coefficient_of_variation']
        
        # Validation: warn if we get unexpected components
        unexpected_components = set(component_cvs.keys()) - expected_components
        if unexpected_components:
            print(f"⚠️  Unexpected components in CV extraction: {unexpected_components}")
            # Remove unexpected components
            for comp in unexpected_components:
                if comp != 'semantic_density':  # Keep semantic_density as fallback
                    del component_cvs[comp]
        
        return component_cvs

def create_evaluation_pipeline(calculator_factory_bounds: Dict[str, Tuple[float, float]], 
                              test_harness_config: Dict[str, Any]) -> Callable:
    """
    Create evaluation pipeline function for Bayesian optimization.
    
    Args:
        calculator_factory_bounds: Parameter bounds from CalculatorFactory
        test_harness_config: Configuration for test harness evaluation
        
    Returns:
        Evaluation function that takes parameter vector and returns reward
    """
    reward_function = DiscriminationRewardFunction()
    
    def evaluate_parameter_vector(parameter_vector: np.ndarray) -> float:
        """
        Evaluate a parameter vector and return composite reward.
        
        Args:
            parameter_vector: Parameter values to evaluate
            
        Returns:
            Composite reward score (higher is better)
        """
        try:
            # Import here to avoid circular dependency
            from calc_integration import CalculatorFactory
            from test_harness import TestConfiguration, SemanticTestHarness
            
            # Convert parameter vector to calculator configuration
            calculator = CalculatorFactory.create_from_vector(
                parameter_vector, 
                embedding_provider=None,  # Will be set by test harness
                base_config_name="baseline"
            )
            
            # Create test configuration
            test_config = TestConfiguration(**test_harness_config)
            
            # Run evaluation
            harness = SemanticTestHarness(test_config)
            results = harness.run_full_experiment()
            
            # Calculate reward
            reward_analysis = reward_function.composite_reward(
                results, 
                parameter_vector, 
                calculator_factory_bounds
            )
            
            return reward_analysis.composite_reward
            
        except Exception as e:
            # Return low reward for failed evaluations
            print(f"Evaluation failed: {e}")
            return 0.0
    
    return evaluate_parameter_vector


def validate_reward_function_with_known_configs():
    """
    Validation function updated for 5-component system.
    """
    reward_func = DiscriminationRewardFunction()
    
    # Test cases with expected behavior (topic_surprise removed)
    test_cases = [
        # Perfect configuration (all in sweet spot) - 5 components
        {
            'name': 'perfect_sweet_spot',
            'cvs': {'ne_density': 0.6, 'conceptual_surprise': 0.5, 'logical_complexity': 0.5, 
                   'conceptual_bridging': 0.6, 'information_density': 0.5},
            'expected_rank': 1  # Should be highest
        },
        
        # Good configuration (most in sweet spot) - 5 components
        {
            'name': 'good_configuration', 
            'cvs': {'ne_density': 0.7, 'conceptual_surprise': 0.6, 'logical_complexity': 0.5, 
                   'conceptual_bridging': 0.3, 'information_density': 0.4},
            'expected_rank': 2
        },
        
        # Current real configuration (from analysis) - 5 components
        {
            'name': 'current_real',
            'cvs': {'ne_density': 0.866, 'conceptual_surprise': 0.592, 'logical_complexity': 0.475, 
                   'conceptual_bridging': 0.608, 'information_density': 0.293},
            'expected_rank': 3
        },
        
        # Poor configuration (some extremes) - 5 components
        {
            'name': 'poor_extremes',
            'cvs': {'ne_density': 1.2, 'conceptual_surprise': 0.1, 'logical_complexity': 0.9, 
                   'conceptual_bridging': 0.05, 'information_density': 1.5},
            'expected_rank': 4
        },
        
        # Broken configuration (all clustering) - 5 components
        {
            'name': 'broken_clustering',
            'cvs': {'ne_density': 0.05, 'conceptual_surprise': 0.02, 'logical_complexity': 0.03, 
                   'conceptual_bridging': 0.02, 'information_density': 0.01},
            'expected_rank': 5  # Should be lowest
        }
    ]
    
    # Calculate scores for all test cases
    results = []
    for test_case in test_cases:
        discrimination_score, cv_penalties = reward_func.calculate_discrimination_reward(test_case['cvs'])
        balance_score, balance_variance, coverage_ratio = reward_func.calculate_balance_reward(test_case['cvs'])
        
        # Dummy parameter vector for stability testing (37 parameters now)
        dummy_params = np.random.random(37) * 0.5 + 0.25  # Middle range values
        dummy_bounds = {f'param_{i}': (0.0, 1.0) for i in range(37)}
        stability_score, _ = reward_func.calculate_stability_penalty(dummy_params, dummy_bounds)
        
        composite = (
            0.40 * discrimination_score +
            0.25 * balance_score +
            0.20 * 1.0 +  # Perfect robustness for single-db test
            0.15 * stability_score
        )
        
        results.append({
            'name': test_case['name'],
            'composite_reward': composite,
            'discrimination': discrimination_score,
            'balance': balance_score,
            'stability': stability_score,
            'expected_rank': test_case['expected_rank']
        })
    
    # Sort by composite reward (highest first)
    results.sort(key=lambda x: x['composite_reward'], reverse=True)
    
    print("🧪 REWARD FUNCTION VALIDATION (5 Components)")
    print("=" * 60)
    print(f"{'Rank':<6} {'Configuration':<20} {'Composite':<12} {'Discrimination':<14} {'Balance':<10}")
    print("-" * 75)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<6} {result['name']:<20} {result['composite_reward']:<12.3f} "
              f"{result['discrimination']:<14.3f} {result['balance']:<10.3f}")
    
    # Check if ranking matches expectations
    ranking_correct = all(
        results[i]['expected_rank'] == i+1 
        for i in range(len(results))
    )
    
    print(f"\n✅ Ranking validation: {'PASSED' if ranking_correct else 'FAILED'}")
    print(f"📊 Testing with {len(test_cases[0]['cvs'])} components (topic_surprise removed)")
    
    return results


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_reward_function_with_known_configs()