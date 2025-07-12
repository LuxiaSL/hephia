"""
Performance optimizations for Bayesian semantic optimizer.
"""

import concurrent.futures
import threading
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
from pathlib import Path


@dataclass
class OptimizedConfig:
    """Enhanced optimization configuration with performance features."""
    
    # Multi-fidelity optimization
    enable_multi_fidelity: bool = True
    exploration_sample_size: int = 4      # Small samples for exploration
    exploitation_sample_size: int = 12    # Large samples for best candidates
    fidelity_threshold: float = 0.6       # Switch to high-fidelity above this reward
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stop_threshold: float = 0.3     # Stop evaluations below this reward
    min_comparisons_before_stop: int = 50 # Minimum comparisons before early stop
    
    # Parallel evaluation
    enable_parallel_evaluation: bool = True
    max_parallel_workers: int = 4         # Number of parallel evaluations
    
    # Surrogate modeling for cheap approximations
    enable_surrogate_modeling: bool = True
    surrogate_sample_ratio: float = 0.3   # 30% of evaluations use surrogate
    
    # Component-specific optimization
    focus_embedding_independent: bool = True  # Prioritize stable components
    stable_component_bonus: float = 1.2      # Bonus weight for stable components


class PerformanceOptimizedEvaluator:
    """
    Optimized evaluator that reduces evaluation cost while maintaining quality.
    """

    def __init__(self, config: OptimizedConfig, base_evaluator: Optional[Callable] = None):
        """
        Initialize performance-optimized evaluator.
        """
        self.config = config
        self.base_evaluator = base_evaluator
        self.evaluation_history = []
        self.surrogate_model = None
        self.executor = None
        
        if config.enable_parallel_evaluation:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=config.max_parallel_workers
            )
    
    def evaluate_with_multi_fidelity(self, parameter_vector: np.ndarray, 
                                    evaluation_id: int, optimizer_instance) -> float:
        """
        Multi-fidelity evaluation: cheap exploration, expensive exploitation.
        """
        
        # Decide fidelity based on evaluation history
        if len(self.evaluation_history) < 10:
            # Always use low fidelity for initial exploration
            fidelity = "low"
        else:
            # Use surrogate model to predict if this is worth high-fidelity evaluation
            predicted_reward = self._predict_reward(parameter_vector)
            fidelity = "high" if predicted_reward > self.config.fidelity_threshold else "low"
        
        if fidelity == "low":
            # Fast evaluation with small sample sizes
            sample_size = self.config.exploration_sample_size
            comparisons = 8  # Reduced comparisons
        else:
            # Full evaluation for promising candidates  
            sample_size = self.config.exploitation_sample_size
            comparisons = 20  # Full comparisons
        
        print(f"   🔬 Eval {evaluation_id}: {fidelity}-fidelity (samples={sample_size})")
        
        # Run evaluation with adaptive sample size
        reward = self._run_adaptive_evaluation(
            parameter_vector, sample_size, comparisons, evaluation_id, optimizer_instance
        )
        
        # Store for surrogate model training
        self.evaluation_history.append({
            'parameter_vector': parameter_vector.copy(),
            'reward': reward,
            'fidelity': fidelity,
            'evaluation_id': evaluation_id,
            'timestamp': time.time()
        })
        
        # Update surrogate model periodically
        if len(self.evaluation_history) % 10 == 0:
            self._update_surrogate_model()
        
        return reward
    
    def _run_adaptive_evaluation(self, parameter_vector: np.ndarray, 
                                      sample_size: int, comparisons: int,
                                      evaluation_id: int, optimizer_instance) -> float:
        """
        Run evaluation with early stopping for poor configurations.
        """
        
        if not self.config.enable_early_stopping:
            base_evaluator = optimizer_instance._create_base_evaluator()
            return base_evaluator(parameter_vector, sample_size, comparisons)
        
        # Start with partial evaluation
        partial_comparisons = min(self.config.min_comparisons_before_stop, comparisons)
        base_evaluator = optimizer_instance._create_base_evaluator()
        partial_reward = base_evaluator(parameter_vector, sample_size, partial_comparisons)
    
        
        # Early stop if clearly poor
        if partial_reward < self.config.early_stop_threshold:
            print(f"      ⏹️ Early stop at {partial_comparisons} comparisons (reward={partial_reward:.3f})")
            return partial_reward
        
        # Continue with full evaluation if promising
        if comparisons > partial_comparisons:
            full_reward = base_evaluator(parameter_vector, sample_size, comparisons)
            return full_reward
        
        return partial_reward
    
    def _predict_reward(self, parameter_vector: np.ndarray) -> float:
        """
        Use simple surrogate model to predict reward without full evaluation.
        """
        
        if not self.evaluation_history or not self.config.enable_surrogate_modeling:
            return 0.5  # Default moderate prediction
        
        # Simple distance-based surrogate (could be upgraded to GP)
        min_distance = float('inf')
        closest_reward = 0.5
        
        for hist in self.evaluation_history[-20:]:  # Use recent history
            distance = np.linalg.norm(parameter_vector - hist['parameter_vector'])
            if distance < min_distance:
                min_distance = distance
                closest_reward = hist['reward']
        
        return closest_reward
    
    def _update_surrogate_model(self):
        """Update surrogate model with recent evaluations."""
        # Placeholder for more sophisticated surrogate modeling
        # Could implement Gaussian Process, Random Forest, etc.
        pass
    
    def evaluate_batch_parallel(self, parameter_vectors: List[np.ndarray]) -> List[float]:
        """
        Evaluate multiple parameter vectors in parallel.
        """
        
        if not self.config.enable_parallel_evaluation or not self.executor:
            return [self.evaluate_with_multi_fidelity(pv, i) for i, pv in enumerate(parameter_vectors)]
        
        print(f"   🔄 Parallel evaluation of {len(parameter_vectors)} configurations...")
        
        # Submit parallel evaluations
        futures = []
        for i, pv in enumerate(parameter_vectors):
            future = self.executor.submit(self.evaluate_with_multi_fidelity, pv, i)
            futures.append(future)
        
        # Collect results
        rewards = []
        for future in concurrent.futures.as_completed(futures):
            try:
                reward = future.result(timeout=300)  # 5 minute timeout per evaluation
                rewards.append(reward)
            except Exception as e:
                print(f"      ❌ Parallel evaluation failed: {e}")
                rewards.append(0.0)  # Failure reward
        
        return rewards
    
    def cleanup(self):
        """Clean up resources safely."""
        try:
            if self.executor:
                print("🔄 Shutting down thread pool...")
                # Give running tasks a chance to complete
                self.executor.shutdown(wait=True, timeout=10)
                print("✅ Thread pool shutdown complete")
        except Exception as e:
            print(f"⚠️  Warning during thread pool cleanup: {e}")
            # Force shutdown if graceful fails
            try:
                if self.executor:
                    self.executor.shutdown(wait=False)
            except:
                pass


# Parameter classification for focused optimization
EMBEDDING_INDEPENDENT_PARAMS = {
    # Amplification factors (raw signal scaling)
    'ne_amplification_factor',           # Entity density thresholds
    'info_density_amplification',        # Information density scaling  
    'concept_surprise_normalization',    # Pattern surprise scaling
    'complexity_normalization',          # Syntactic complexity scaling
    'topic_surprise_amplification',      # Topic surprise scaling
    
    # Discriminator weights (spacy-based features)
    'dependency_weight', 'pos_weight', 'semantic_reasoning_weight',
    'technical_info_weight', 'social_info_weight', 'factual_info_weight',
    'syntactic_bridge_weight', 'semantic_bridge_weight', 'entity_bridge_weight',
    'syntactic_surprise_weight', 'semantic_role_surprise_weight', 'discourse_surprise_weight',
    'topic_discontinuity_weight', 'density_surprise_weight', 'structure_surprise_weight',
    
    # Transformation parameters (curve shaping)
    'low_power', 'high_scale', 'mid_slope', 'low_threshold', 'mid_threshold',
    'low_scale', 'high_base', 'min_output', 'max_output',
    
    # Normalization factors
    'bridge_normalization', 'info_normalization', 'topic_surprise_normalization',
    
    # Configuration parameters
    'abstract_boost', 'concrete_boost', 'length_weight',
    'min_sentences_for_cohesion', 'min_sentences_for_topic_surprise',
}

EMBEDDING_DEPENDENT_PARAMS = {
    # Component weights (need rebalancing per provider)
    'topic_surprise_weight', 'ne_density_weight', 'conceptual_surprise_weight',
    'logical_complexity_weight', 'conceptual_bridging_weight', 'information_density_weight',
    
    # Embedding-specific thresholds (only if using embedding_similarity method)
    'cohesion_fallback_similarity',
}


def get_focused_parameter_bounds():
    """
    Get parameter bounds focused on embedding-independent parameters.
    These are most likely to transfer between embedding providers.
    """
    from calc_integration import CalculatorFactory
    
    all_bounds = CalculatorFactory.get_parameter_bounds()
    
    # Focus on embedding-independent parameters + component weights
    focused_bounds = {}
    
    for param_name, bounds in all_bounds.items():
        if (param_name in EMBEDDING_INDEPENDENT_PARAMS or 
            param_name in EMBEDDING_DEPENDENT_PARAMS):
            focused_bounds[param_name] = bounds
    
    print(f"🎯 FOCUSED PARAMETER OPTIMIZATION")
    print(f"   Original space: {len(all_bounds)} dimensions")
    print(f"   Focused space: {len(focused_bounds)} dimensions")
    print(f"   Embedding-independent: {len(EMBEDDING_INDEPENDENT_PARAMS)}")
    print(f"   Component weights: {len(EMBEDDING_DEPENDENT_PARAMS)}")
    
    return focused_bounds


def create_production_optimizer_config(
    n_calls: int = 100,
    enable_all_optimizations: bool = True,
    focus_parameters: bool = True
) -> Dict[str, Any]:
    """
    Create production-ready optimizer configuration.
    """
    
    config = {
        'n_calls': n_calls,
        'sample_size': 8 if enable_all_optimizations else 10,
        'comparisons_per_sample': 16 if enable_all_optimizations else 20,
        
        # Performance optimizations
        'enable_multi_fidelity': enable_all_optimizations,
        'enable_early_stopping': enable_all_optimizations,  
        'enable_parallel_evaluation': enable_all_optimizations,
        'max_parallel_workers': 4 if enable_all_optimizations else 1,
    }
    
    if enable_all_optimizations:
        config['n_initial_points'] = max(5, n_calls // 15)  # Fewer random points
    else:
        config['n_initial_points'] = max(10, n_calls // 8)  # Standard random points
    
    # Parameter space selection
    if focus_parameters:
        config['parameter_bounds'] = get_focused_parameter_bounds()
        config['focus_embedding_independent'] = True
    
    return config


def load_hybrid_seed_config(seed_path: str) -> Optional[Dict[str, Any]]:
    """Load hybrid seed configuration for warm-start optimization."""
    
    if not Path(seed_path).exists():
        print(f"⚠️ Hybrid seed file not found: {seed_path}")
        return None
    
    try:
        with open(seed_path, 'r') as f:
            hybrid_config = json.load(f)
        
        print(f"🧬 Loaded hybrid seed from: {seed_path}")
        
        if 'expected_cvs' in hybrid_config:
            print(f"   Expected component improvements:")
            for comp, cv in hybrid_config['expected_cvs'].items():
                print(f"     {comp:20} CV≥{cv:.3f}")
        
        return hybrid_config
        
    except Exception as e:
        print(f"❌ Failed to load hybrid seed: {e}")
        return None


def estimate_optimization_speedup(n_calls: int = 100, enable_optimizations: bool = True):
    """
    Estimate performance improvement from optimizations.
    """
    
    print(f"📊 OPTIMIZATION PERFORMANCE ANALYSIS")
    print(f"=" * 50)
    
    # Current system timing estimates (from your actual results)
    current_eval_time = 3.0  # minutes per evaluation
    current_total_time = n_calls * current_eval_time
    
    if not enable_optimizations:
        print(f"   Current system: {current_total_time:.0f} minutes ({current_total_time/60:.1f} hours)")
        return
    
    # Optimized system estimates
    speedup_factors = {
        'multi_fidelity': 0.4,      # 60% of evals use low-fidelity (4x faster)
        'early_stopping': 0.7,      # 30% reduction from early stops
        'parallel_evaluation': 0.5,  # 2x speedup from parallelism
        'focused_parameters': 0.8,   # 20% reduction from smaller space
    }
    
    optimized_eval_time = current_eval_time
    for optimization, factor in speedup_factors.items():
        optimized_eval_time *= factor
        print(f"   {optimization:20}: {factor:.1f}x → {optimized_eval_time:.1f} min/eval")
    
    optimized_total_time = n_calls * optimized_eval_time
    total_speedup = current_total_time / optimized_total_time
    
    print(f"\n🚀 TOTAL PERFORMANCE IMPROVEMENT:")
    print(f"   Current system: {current_total_time:.0f} minutes ({current_total_time/60:.1f} hours)")
    print(f"   Optimized system: {optimized_total_time:.0f} minutes ({optimized_total_time/60:.1f} hours)")
    print(f"   Speedup: {total_speedup:.1f}x faster")
    print(f"   Time saved: {(current_total_time - optimized_total_time)/60:.1f} hours")


if __name__ == "__main__":
    # Show performance analysis
    print("🔥 PERFORMANCE OPTIMIZATION PREVIEW")
    print("=" * 50)
    estimate_optimization_speedup(100, True)
    
    # Show focused parameter space
    print(f"\n")
    focused_bounds = get_focused_parameter_bounds()
    print(f"\n💡 Key embedding-independent parameters:")
    for param in sorted(list(EMBEDDING_INDEPENDENT_PARAMS)[:10]):  # Show first 10
        print(f"   {param}")
    print(f"   ... and {len(EMBEDDING_INDEPENDENT_PARAMS)-10} more")