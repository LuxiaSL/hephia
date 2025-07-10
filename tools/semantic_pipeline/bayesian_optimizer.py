#!/usr/bin/env python3
"""
bayesian_optimizer.py

Bayesian optimization engine for neural semantic discrimination parameter tuning.
Uses scikit-optimize to find optimal parameter configurations that maximize
neuromorphic memory discrimination across multiple cognitive dimensions.

Based on the reward function design philosophy:
- High compute budget (300-500 evaluations)
- No early stopping to find true optima
- Multiple independent runs to discover different optimal patterns
- Comprehensive experiment tracking and analysis
"""

import os
import time
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import traceback

# Scikit-optimize imports
try:
    from skopt import gp_minimize, forest_minimize, dummy_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
    SKOPT_AVAILABLE = True
except ImportError:
    print("Warning: scikit-optimize not available. Install with: pip install scikit-optimize")
    SKOPT_AVAILABLE = False

# Local imports
from reward_functions import DiscriminationRewardFunction, RewardAnalysis
from calc_integration import CalculatorFactory
from test_harness import TestConfiguration, SemanticTestHarness


@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization runs."""
    
    # Optimization parameters
    n_calls: int = 300                    # Number of evaluations (high compute budget)
    n_initial_points: int = 20            # Random exploration before Bayesian
    acquisition_function: str = 'EI'      # Expected Improvement
    n_jobs: int = 1                       # Parallel evaluations (1 for now)
    random_state: int = 42               # For reproducible results
    
    # Test harness configuration
    sample_size: int = 10                # Number of sample nodes per evaluation
    comparisons_per_sample: int = 50     # Comparisons per sample
    database_paths: List[str] = None     # Paths to memory databases
    embedding_providers: List[str] = None # Embedding providers to test
    
    # Experiment tracking
    experiment_name: str = "neural_optimization"
    output_dir: str = "optimization_results"
    save_frequency: int = 10             # Save every N evaluations
    
    # Termination criteria (loose to find true optima)
    max_no_improvement: int = 100        # Allow long plateaus
    target_reward: float = 0.95          # Stop if we achieve near-perfect reward
    
    def __post_init__(self):
        """Set defaults and validate configuration."""
        if self.database_paths is None:
            # Try to find databases automatically
            possible_paths = [
                "dbs/cypher.db",
                "dbs/haiku.db",
                "dbs/sonnet.db",
            ]
            existing_paths = [p for p in possible_paths if Path(p).exists()]
            self.database_paths = ["dbs/cypher.db", "dbs/haiku.db", "dbs/sonnet.db"]
        
        if self.embedding_providers is None:
            self.embedding_providers = ["stella"]  # Real embedding provider
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)


@dataclass
class OptimizationResult:
    """Single evaluation result for tracking."""
    
    evaluation_id: int
    parameter_vector: np.ndarray
    reward_analysis: RewardAnalysis
    execution_time: float
    timestamp: datetime
    experiment_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'evaluation_id': self.evaluation_id,
            'parameter_vector': self.parameter_vector.tolist(),
            'reward_analysis': self.reward_analysis.to_dict(),
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'experiment_metadata': self.experiment_metadata
        }


class BayesianSemanticOptimizer:
    """
    Bayesian optimization engine for semantic discrimination parameters.
    
    Finds optimal parameter configurations using Gaussian Process optimization
    with the neuromorphic memory discrimination reward function.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize Bayesian optimizer.
        
        Args:
            config: Optimization configuration
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize required. Install with: pip install scikit-optimize")
        
        self.config = config
        self.reward_function = DiscriminationRewardFunction()
        
        # Get parameter space from calculator factory
        self.parameter_bounds = CalculatorFactory.get_parameter_bounds()
        self.parameter_names = list(self.parameter_bounds.keys())
        self.search_space = self._create_search_space()
        
        # Experiment tracking
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        self.start_time = None
        
        # Create experiment directory - respect custom output dirs
        if hasattr(self.config, '_custom_output_dir') and self.config._custom_output_dir:
            # Use exact custom directory specified by user
            
            base_dir = Path("optimization_results")
        if self.config.output_dir.startswith("optimization_results"):
            # Already has base directory
            self.experiment_dir = Path(self.config.output_dir)
        else:
            # Put under optimization_results/
            self.experiment_dir = base_dir / self.config.output_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧬 Initialized Bayesian optimizer")
        print(f"   Parameter space: {len(self.parameter_names)} dimensions")
        print(f"   Optimization budget: {self.config.n_calls} evaluations")
        print(f"   Experiment directory: {self.experiment_dir}")
    
    def _create_search_space(self) -> List:
        """Create scikit-optimize search space from parameter bounds."""
        space = []
        
        for param_name in self.parameter_names:
            min_val, max_val = self.parameter_bounds[param_name]
            
            # Determine if this should be integer or real
            if 'min_sentences' in param_name or 'max_workers' in param_name:
                space.append(Integer(int(min_val), int(max_val), name=param_name))
            else:
                space.append(Real(min_val, max_val, name=param_name))
        
        return space
    
    def _evaluate_parameter_vector(self, parameter_vector: np.ndarray) -> float:
        """
        Evaluate a parameter vector and return composite reward.
        
        This is the core objective function for Bayesian optimization.
        
        Args:
            parameter_vector: Parameter values to evaluate
            
        Returns:
            Composite reward score (higher is better)
        """
        evaluation_start = time.time()
        evaluation_id = len(self.results) + 1
        
        try:
            # Progress tracked by logger
            if evaluation_id == 1 or evaluation_id % 10 == 0:  # Less frequent updates
                print(f"🔬 Evaluation {evaluation_id}/{self.config.n_calls}")
            
            # Convert parameter vector to calculator configuration
            calculator = CalculatorFactory.create_from_vector(
                parameter_vector,
                embedding_provider=None,  # Will be set by test harness
                base_config_name="baseline"
            )
            
            # Create test configuration for this evaluation
            test_config = TestConfiguration(
                sample_size=self.config.sample_size,
                comparisons_per_sample=self.config.comparisons_per_sample,
                database_paths=self.config.database_paths,
                embedding_providers=self.config.embedding_providers,
                calculator_configs=["custom"],  # We'll provide the calculator directly
                output_dir=str(self.experiment_dir / f"eval_{evaluation_id}"),
            )
            
            # Run semantic test harness evaluation
            harness = SemanticTestHarness(test_config)
            
            # Inject our custom calculator
            original_create_calculator = harness._create_calculator
            def custom_create_calculator(calc_config, embedding_provider):
                return calculator  # Use our optimized calculator
            harness._create_calculator = custom_create_calculator
                    
            experiment_results = harness.run_full_experiment()
            
            # Calculate reward using our reward function
            reward_analysis = self.reward_function.composite_reward(
                experiment_results,
                parameter_vector,
                self.parameter_bounds
            )
            
            execution_time = time.time() - evaluation_start
            
            # Create result record
            result = OptimizationResult(
                evaluation_id=evaluation_id,
                parameter_vector=parameter_vector.copy(),
                reward_analysis=reward_analysis,
                execution_time=execution_time,
                timestamp=datetime.now(),
                experiment_metadata={
                    'test_config': asdict(test_config),
                    'total_comparisons': experiment_results.get('analysis_summary', {}).get('total_comparisons', 0),
                    'component_cvs': reward_analysis.component_cvs
                }
            )
            
            # Track result
            self.results.append(result)
            
            # Update best result
            if self.best_result is None or reward_analysis.composite_reward > self.best_result.reward_analysis.composite_reward:
                self.best_result = result
                print(f"🏆 NEW BEST: {reward_analysis.composite_reward:.4f}")
            
            # Save progress periodically
            if evaluation_id % self.config.save_frequency == 0:
                self._save_progress()
            
            if reward_analysis.composite_reward > (self.best_result.reward_analysis.composite_reward if self.best_result else 0):
                print(f"🏆 NEW BEST: {reward_analysis.composite_reward:.4f}")
                for comp, cv in reward_analysis.component_cvs.items():
                    print(f"     {comp}: {cv:.3f}")
            
            return reward_analysis.composite_reward
            
        except Exception as e:
            print(f"❌ Evaluation {evaluation_id} failed: {e}")
            print(traceback.format_exc())
            
            # Record failed evaluation
            failed_result = OptimizationResult(
                evaluation_id=evaluation_id,
                parameter_vector=parameter_vector.copy(),
                reward_analysis=RewardAnalysis(0.0, 0.0, 0.0, 0.0, 0.0, {}, {}, 0.0, 0.0, []),
                execution_time=time.time() - evaluation_start,
                timestamp=datetime.now(),
                experiment_metadata={'error': str(e)}
            )
            self.results.append(failed_result)
            
            # Return low reward for failed evaluations
            return 0.0
    
    def optimize(self) -> OptimizationResult:
        """
        Run Bayesian optimization to find optimal parameters.
        
        Returns:
            Best result found during optimization
        """
        print(f"🚀 Starting Bayesian optimization")
        print(f"   Budget: {self.config.n_calls} evaluations")
        print(f"   Initial random: {self.config.n_initial_points}")
        print(f"   Acquisition: {self.config.acquisition_function}")
        
        self.start_time = time.time()
        
        # Create objective function with named parameters
        @use_named_args(self.search_space)
        def objective(**params):
            # Convert named parameters to vector
            parameter_vector = np.array([params[name] for name in self.parameter_names])
            return -self._evaluate_parameter_vector(parameter_vector)  # Minimize negative reward
        
        try:
            # Run Bayesian optimization
            optimization_result = gp_minimize(
                func=objective,
                dimensions=self.search_space,
                n_calls=self.config.n_calls,
                n_initial_points=self.config.n_initial_points,
                acq_func=self.config.acquisition_function,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=False  # We handle our own progress output
            )
            
            # Final save
            self._save_progress()
            # Skip pickle save to avoid local function issues
            
            total_time = time.time() - self.start_time
            print(f"\n🏁 Optimization complete!")
            print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"   Evaluations: {len(self.results)}")
            print(f"   Best reward: {self.best_result.reward_analysis.composite_reward:.4f}")
            
            return self.best_result
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            print(traceback.format_exc())
            
            # Save what we have
            self._save_progress()
            
            return self.best_result
    
    def _save_progress(self):
        """Save current optimization progress."""
        progress_file = self.experiment_dir / "optimization_progress.json"
        
        progress_data = {
            'config': asdict(self.config),
            'parameter_bounds': self.parameter_bounds,
            'parameter_names': self.parameter_names,
            'num_evaluations': len(self.results),
            'best_reward': self.best_result.reward_analysis.composite_reward if self.best_result else 0.0,
            'results': [result.to_dict() for result in self.results],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"💾 Progress saved: {len(self.results)} evaluations")
    
    def _save_optimization_result(self, skopt_result):
        """Save the final scikit-optimize result object."""
        result_file = self.experiment_dir / "skopt_result.pkl"
        
        with open(result_file, 'wb') as f:
            pickle.dump(skopt_result, f)
        
        # Also save human-readable summary
        summary_file = self.experiment_dir / "optimization_summary.json"
        
        summary = {
            'best_parameters': {
                name: float(val) for name, val in zip(self.parameter_names, skopt_result.x)
            },
            'best_score': float(-skopt_result.fun),  # Convert back from minimization
            'convergence_info': {
                'n_calls': len(skopt_result.y_iters),
                'best_iteration': int(np.argmin(skopt_result.y_iters)),
                'final_improvement': float(skopt_result.y_iters[0] - skopt_result.fun)
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def run_optimization_experiment(
    experiment_name: str = "neural_optimization_test",
    n_calls: int = 50,  # Reduced for testing
    database_paths: List[str] = None,
    sample_size: int = 5,
    comparisons_per_sample: int = 20
) -> OptimizationResult:
    """
    Convenience function to run a complete optimization experiment.
    
    Args:
        experiment_name: Name for the experiment
        n_calls: Number of optimization evaluations
        database_paths: Paths to memory databases
        sample_size: Number of sample nodes per evaluation  
        comparisons_per_sample: Comparisons per sample
        
    Returns:
        Best optimization result
    """
    config = OptimizationConfig(
        experiment_name=experiment_name,
        n_calls=n_calls,
        database_paths=database_paths,
        sample_size=sample_size,
        comparisons_per_sample=comparisons_per_sample,
        n_initial_points=min(10, n_calls // 5),  # 20% random exploration
    )
    
    optimizer = BayesianSemanticOptimizer(config)
    return optimizer.optimize()


def quick_optimization_test():
    """Run a quick optimization test to validate the system."""
    print("🧪 RUNNING QUICK OPTIMIZATION TEST")
    print("=" * 50)
    
    # Quick test with minimal parameters
    result = run_optimization_experiment(
        experiment_name="quick_test",
        n_calls=10,  # Very small for testing
        sample_size=3,
        comparisons_per_sample=10
    )
    
    if result:
        print(f"\n✅ Test completed successfully!")
        print(f"Best reward: {result.reward_analysis.composite_reward:.4f}")
        print(f"Component CVs: {result.reward_analysis.component_cvs}")
    else:
        print(f"\n❌ Test failed!")
    
    return result


if __name__ == "__main__":
    # Run quick test when script is executed directly
    quick_optimization_test()