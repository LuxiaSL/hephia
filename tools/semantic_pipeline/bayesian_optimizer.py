#!/usr/bin/env python3
"""
bayesian_optimizer.py

Clean Bayesian optimization engine for semantic discrimination parameter tuning.

Updated to support:
- Performance optimizations (multi-fidelity, early stopping, parallelization)
- Hybrid seeding from previous optimization runs
- Focused parameter space optimization
"""

import os
import time
import json
import pickle
import signal
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass, asdict
import traceback

# Scikit-optimize imports
try:
    from skopt import gp_minimize, forest_minimize, dummy_minimize, Optimizer
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
    random_state: int = int(time.time()) % 10000
    
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
    
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None  # Custom parameter bounds
    hybrid_seed_config: Optional[Dict[str, Any]] = None               # Hybrid seed configuration
    verbose: bool = True                                              # Clean logging control
    
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
            self.database_paths = existing_paths if existing_paths else ["dbs/cypher.db", "dbs/haiku.db", "dbs/sonnet.db"]
        
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


class CleanBayesianSemanticOptimizer:
    """
    Clean Bayesian optimization engine with integrated performance features.
    Includes multi-fidelity evaluation, hybrid seeding, and focused parameter optimization.
    """
    
    def __init__(self, config: OptimizationConfig, performance_evaluator: Optional[Any] = None, use_focused_bounds: bool = True):
        """
        Initialize clean Bayesian optimizer.
        
        Args:
            config: Optimization configuration
            performance_evaluator: Optional PerformanceOptimizedEvaluator for enhanced features
            use_focused_bounds: Whether to use focused parameter bounds based on analysis
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize required. Install with: pip install scikit-optimize")
        
        self.config = config
        self.performance_evaluator = performance_evaluator
        self.reward_function = DiscriminationRewardFunction()
        
        # Set up parameter space with FOCUSED BOUNDS option
        if config.parameter_bounds:
            self.parameter_bounds = config.parameter_bounds
        elif use_focused_bounds:
            self.parameter_bounds = CalculatorFactory.get_focused_parameter_bounds()
            if self.config.verbose:
                print("🎯 Using focused parameter bounds based on statistical analysis")
        else:
            self.parameter_bounds = CalculatorFactory.get_parameter_bounds()
            if self.config.verbose:
                print("📊 Using standard parameter bounds")
            
        self.parameter_names = list(self.parameter_bounds.keys())
        self.search_space = self._create_search_space()
        
        expected_param_count = 37
        if len(self.parameter_names) != expected_param_count:
            if self.config.verbose:
                print(f"⚠️  Parameter count mismatch: got {len(self.parameter_names)}, expected {expected_param_count}")
                print(f"    This might indicate missing or extra parameters in bounds")

        # Experiment tracking
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        self.start_time = None
        
        # Create experiment directory
        self.experiment_dir = Path(config.output_dir)
        if not str(config.output_dir).endswith(('/', '\\')):
            # Auto-generate unique directory if not specified
            self.experiment_dir = self.experiment_dir / f"{config.experiment_name}_{int(time.time())}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scikit-optimize optimizer for ask-tell interface
        self.optimizer = None
        
        if config.verbose:
            print(f"🧬 Initialized clean Bayesian optimizer")
            print(f"   Parameter space: {len(self.parameter_names)} dimensions")
            print(f"   Optimization budget: {config.n_calls} evaluations")
            print(f"   Performance features: {'enabled' if performance_evaluator else 'disabled'}")
            print(f"   Hybrid seed: {'loaded' if config.hybrid_seed_config else 'none'}")
            print(f"   Experiment directory: {self.experiment_dir}")
    
    def _create_search_space(self) -> List:
        """Create scikit-optimize search space with proper type detection."""
        space = []
        
        for param_name in self.parameter_names:
            min_val, max_val = self.parameter_bounds[param_name]
            
            # CLEAN type detection (no string matching hacks)
            if self._is_integer_parameter(param_name):
                space.append(Integer(int(min_val), int(max_val), name=param_name))
            else:
                space.append(Real(min_val, max_val, name=param_name))
        
        return space
    
    def _is_integer_parameter(self, param_name: str) -> bool:
        """Determine if parameter should be integer type."""
        integer_indicators = [
            'min_sentences', 'max_workers', 'cache_size',
            '_for_cohesion', '_for_topic_surprise'
        ]
        return any(indicator in param_name for indicator in integer_indicators)
    
    def _create_base_evaluator(self) -> Callable:
        """
        Create base evaluation function with clean dependency injection.
        """
        
        def evaluate_parameter_vector(parameter_vector: np.ndarray, 
                                    sample_size: Optional[int] = None,
                                    comparisons_per_sample: Optional[int] = None) -> float:
            """
            Clean evaluation function with dependency injection.
            """
            evaluation_start = time.time()
            evaluation_id = len(self.results) + 1
            
            try:
                # Use provided sample sizes or defaults from config
                eval_sample_size = sample_size or self.config.sample_size
                eval_comparisons = comparisons_per_sample or self.config.comparisons_per_sample
                
                if self.config.verbose and (evaluation_id == 1 or evaluation_id % 10 == 0):
                    print(f"🔬 Evaluation {evaluation_id}/{self.config.n_calls}")
                
                # Create calculator from parameter vector (CLEAN DEPENDENCY INJECTION)
                calculator = CalculatorFactory.create_from_vector(
                    parameter_vector,
                    embedding_provider=None,  # Will be set by test harness
                    base_config_name="baseline"
                )
                
                # Create test configuration for this evaluation
                test_config = TestConfiguration(
                    sample_size=eval_sample_size,
                    comparisons_per_sample=eval_comparisons,
                    database_paths=self.config.database_paths,
                    embedding_providers=self.config.embedding_providers,
                    calculator_configs=["custom"],  # We'll inject the calculator directly
                    output_dir=str(self.experiment_dir / f"eval_{evaluation_id}"),
                    generate_detailed_logs=False  # Skip logs during optimization
                )
                
                harness = SemanticTestHarness(test_config)
                
                harness.injected_calculator = calculator
                
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
                        'component_cvs': reward_analysis.component_cvs,
                        'sample_size': eval_sample_size,
                        'comparisons_per_sample': eval_comparisons
                    }
                )
                
                # Track result
                self.results.append(result)
                
                # Update best result
                if self.best_result is None or reward_analysis.composite_reward > self.best_result.reward_analysis.composite_reward:
                    self.best_result = result
                    if self.config.verbose:
                        print(f"🏆 NEW BEST: {reward_analysis.composite_reward:.4f}")
                        for comp, cv in reward_analysis.component_cvs.items():
                            print(f"     {comp}: {cv:.3f}")
                
                # Save progress periodically
                if evaluation_id % self.config.save_frequency == 0:
                    self._save_progress()
                
                return reward_analysis.composite_reward
                
            except Exception as e:
                if self.config.verbose:
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
        
        return evaluate_parameter_vector
    
    def _validate_and_load_hybrid_seed(self, hybrid_seed_config: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Robust hybrid seed validation and loading with comprehensive error handling.
        
        Returns:
            Valid parameter vector or None if hybrid seed is invalid
        """
        if not hybrid_seed_config:
            return None
        
        try:
            # Check for required fields
            if 'parameter_vector' not in hybrid_seed_config:
                if self.config.verbose:
                    print("⚠️  Hybrid seed missing 'parameter_vector', skipping")
                return None
            
            raw_vector = hybrid_seed_config['parameter_vector']
            
            # Validate it's a list/array of numbers
            if not isinstance(raw_vector, (list, np.ndarray)):
                if self.config.verbose:
                    print("⚠️  Hybrid seed parameter_vector is not a list/array, skipping")
                return None
            
            # Convert to numpy array
            try:
                parameter_vector = np.array(raw_vector, dtype=float)
            except (ValueError, TypeError):
                if self.config.verbose:
                    print("⚠️  Hybrid seed parameter_vector contains non-numeric values, skipping")
                return None
            
            # Check vector length matches current parameter space
            expected_length = len(self.parameter_names)
            if len(parameter_vector) != expected_length:
                if self.config.verbose:
                    print(f"⚠️  Hybrid seed length {len(parameter_vector)} doesn't match current parameter space {expected_length}")
                    print(f"    This usually means the component structure changed. Skipping hybrid seed.")
                return None
            
            # Validate parameter bounds (but allow small violations)
            bounds_violations = []
            for i, (param_name, param_value) in enumerate(zip(self.parameter_names, parameter_vector)):
                if param_name in self.parameter_bounds:
                    min_val, max_val = self.parameter_bounds[param_name]
                    violation_threshold = (max_val - min_val) * 0.1  # Allow 10% violation
                    
                    if param_value < (min_val - violation_threshold) or param_value > (max_val + violation_threshold):
                        bounds_violations.append(f"{param_name}: {param_value:.3f} outside [{min_val:.3f}, {max_val:.3f}]")
            
            if bounds_violations and len(bounds_violations) > len(parameter_vector) * 0.3:  # If >30% parameters are badly out of bounds
                if self.config.verbose:
                    print(f"⚠️  Too many parameter bound violations ({len(bounds_violations)}/{len(parameter_vector)})")
                    print(f"    This suggests incompatible parameter space. Skipping hybrid seed.")
                return None
            elif bounds_violations and self.config.verbose:
                print(f"📋 Hybrid seed has {len(bounds_violations)} parameter bound violations (within tolerance)")
                for violation in bounds_violations[:3]:  # Show first 3
                    print(f"    {violation}")
                if len(bounds_violations) > 3:
                    print(f"    ... and {len(bounds_violations) - 3} more")
            
            # Test if we can actually create a calculator from this vector
            try:
                test_calculator = CalculatorFactory.create_from_vector(
                    parameter_vector,
                    embedding_provider=None,
                    base_config_name="baseline"
                )
                if self.config.verbose:
                    print("✅ Hybrid seed validation successful")
                    
                    # Show expected CVs if available
                    if 'expected_cvs' in hybrid_seed_config:
                        expected_cvs = hybrid_seed_config['expected_cvs']
                        print(f"🎯 Expected performance:")
                        for comp, cv in expected_cvs.items():
                            if comp != 'topic_surprise':  # Skip the removed component
                                print(f"     {comp}: CV≥{cv:.3f}")
                
                return parameter_vector
                
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Hybrid seed fails calculator creation: {e}")
                return None
        
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Hybrid seed validation failed: {e}")
            return None
    
    def optimize(self) -> OptimizationResult:
        """
        Run Bayesian optimization with clean architecture and performance features.
        
        Returns:
            Best result found during optimization
        """
        self._setup_signal_handlers()

        if self.config.verbose:
            print(f"🚀 Starting clean Bayesian optimization")
            print(f"   Budget: {self.config.n_calls} evaluations")
            print(f"   Initial random: {self.config.n_initial_points}")
            print(f"   Acquisition: {self.config.acquisition_function}")
        
        self.start_time = time.time()
        
        try:
            if self.performance_evaluator:
                # Use performance-optimized evaluation with ask-tell interface
                return self._optimize_with_performance_features()
            else:
                # Use standard optimization
                return self._optimize_standard()
                
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Optimization failed: {e}")
                print(traceback.format_exc())
            
            # Save what we have
            self._save_progress()
            return self.best_result
    
    def _optimize_standard(self) -> OptimizationResult:
        """Standard optimization with SAFE hybrid seed handling."""
        self._setup_signal_handlers()
        
        # Create objective function with named parameters
        @use_named_args(self.search_space)
        def objective(**params):
            # Convert named parameters to vector
            parameter_vector = np.array([params[name] for name in self.parameter_names])
            base_evaluator = self._create_base_evaluator()
            return -base_evaluator(parameter_vector)  # Minimize negative reward
        
        # SAFE hybrid seed loading
        initial_points = []
        if self.config.hybrid_seed_config:
            hybrid_vector = self._validate_and_load_hybrid_seed(self.config.hybrid_seed_config)
            if hybrid_vector is not None:
                initial_points = [hybrid_vector.tolist()]  # Convert to list for scikit-optimize
                if self.config.verbose:
                    print(f"🎯 Starting optimization with validated hybrid seed")
            else:
                if self.config.verbose:
                    print(f"🔄 Hybrid seed invalid, starting with random initialization")
        
        # Run Bayesian optimization
        try:
            optimization_result = gp_minimize(
                func=objective,
                dimensions=self.search_space,
                n_calls=self.config.n_calls,
                n_initial_points=self.config.n_initial_points,
                x0=initial_points if initial_points else None,  # Safe hybrid seed usage
                acq_func=self.config.acquisition_function,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=False  # We handle our own progress output
            )
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Optimization failed: {e}")
                if initial_points:
                    print(f"🔄 Retrying without hybrid seed...")
                    # Retry without hybrid seed
                    optimization_result = gp_minimize(
                        func=objective,
                        dimensions=self.search_space,
                        n_calls=self.config.n_calls,
                        n_initial_points=self.config.n_initial_points,
                        x0=None,  # No hybrid seed
                        acq_func=self.config.acquisition_function,
                        n_jobs=self.config.n_jobs,
                        random_state=self.config.random_state,
                        verbose=False
                    )
                else:
                    raise
        
        # Final save
        self._save_progress()
        
        total_time = time.time() - self.start_time
        if self.config.verbose:
            print(f"\n🏁 Standard optimization complete!")
            print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"   Evaluations: {len(self.results)}")
            if self.best_result:
                print(f"   Best reward: {self.best_result.reward_analysis.composite_reward:.4f}")
        
        return self.best_result
    
    def _optimize_with_performance_features(self) -> OptimizationResult:
        """Optimization with performance features using ask-tell interface."""
        self._setup_signal_handlers()
        
        if self.config.verbose:
            print(f"🔥 Using performance-optimized evaluation")
        
        # Initialize optimizer for ask-tell interface
        self.optimizer = Optimizer(
            dimensions=self.search_space,
            acq_func=self.config.acquisition_function,
            n_initial_points=self.config.n_initial_points,
            random_state=self.config.random_state
        )
        
        # Add hybrid seed as initial point if available
        if self.config.hybrid_seed_config and 'parameter_vector' in self.config.hybrid_seed_config:
            hybrid_vector = self.config.hybrid_seed_config['parameter_vector']
            if len(hybrid_vector) == len(self.parameter_names):
                if self.config.verbose:
                    print(f"🎯 Starting with hybrid seed")
                hybrid_reward = self.performance_evaluator.evaluate_with_multi_fidelity(
                    np.array(hybrid_vector), len(self.results) + 1, self
                )
                self.optimizer.tell([hybrid_vector], [-hybrid_reward])
        
        # Main optimization loop with ask-tell interface
        for iteration in range(self.config.n_calls - len(self.results)):
            current_eval_id = len(self.results) + 1

            if self.config.verbose and iteration % 20 == 0:
                print(f"🔄 Optimization iteration {current_eval_id}/{self.config.n_calls}")
            
            # Ask for next point(s) to evaluate
            if (hasattr(self.performance_evaluator, 'config') and 
                self.performance_evaluator.config.enable_parallel_evaluation):
                # Parallel evaluation
                n_points = min(3, getattr(self.performance_evaluator.config, 'max_parallel_workers', 2))
                next_points = self.optimizer.ask(n_points=n_points)
                
                # Evaluate points in parallel using performance evaluator
                rewards = []
                for i, point in enumerate(next_points):
                    point_eval_id = current_eval_id + i
                    reward = self.performance_evaluator.evaluate_with_multi_fidelity(
                        np.array(point), point_eval_id, self
                    )
                    rewards.append(reward)
                
                # Tell optimizer the results
                self.optimizer.tell(next_points, [-r for r in rewards])  # Negative for minimization
                
            else:
                # Sequential evaluation
                next_point = self.optimizer.ask()
                reward = self.performance_evaluator.evaluate_with_multi_fidelity(
                    np.array(next_point), current_eval_id, self
                )
                self.optimizer.tell([next_point], [-reward])  # Negative for minimization
            
            # Check for early termination
            if (self.best_result and 
                self.best_result.reward_analysis.composite_reward > self.config.target_reward):
                if self.config.verbose:
                    print(f"🎯 Target reward achieved: {self.best_result.reward_analysis.composite_reward:.4f}")
                break
        
        # Final save
        self._save_progress()
        
        total_time = time.time() - self.start_time
        if self.config.verbose:
            print(f"\n🏁 Performance-optimized optimization complete!")
            print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"   Evaluations: {len(self.results)}")
            if self.best_result:
                print(f"   Best reward: {self.best_result.reward_analysis.composite_reward:.4f}")
        
        return self.best_result
    
    def _save_progress(self):
        """Save current optimization progress."""
        progress_file = self.experiment_dir / "optimization_progress.json"

        # Calculate current runtime
        current_time = time.time()
        runtime = current_time - self.start_time if self.start_time else 0
        
        progress_data = {
            'config': asdict(self.config),
            'parameter_bounds': self.parameter_bounds,
            'parameter_names': self.parameter_names,
            'num_evaluations': len(self.results),
            'best_reward': self.best_result.reward_analysis.composite_reward if self.best_result else 0.0,
            'results': [result.to_dict() for result in self.results],
            'performance_features': {
                'enabled': self.performance_evaluator is not None,
                'multi_fidelity': getattr(self.performance_evaluator, 'config', None) and self.performance_evaluator.config.enable_multi_fidelity if self.performance_evaluator else False,
                'early_stopping': getattr(self.performance_evaluator, 'config', None) and self.performance_evaluator.config.enable_early_stopping if self.performance_evaluator else False,
                'parallel_evaluation': getattr(self.performance_evaluator, 'config', None) and self.performance_evaluator.config.enable_parallel_evaluation if self.performance_evaluator else False,
            },
            'hybrid_seed_used': self.config.hybrid_seed_config is not None,
            'runtime_info': {
                'total_runtime_seconds': runtime,
                'evaluations_per_minute': (len(self.results) / (runtime / 60)) if runtime > 0 else 0,
                'estimated_completion_time': (self.config.n_calls - len(self.results)) * (runtime / len(self.results)) if len(self.results) > 0 else 0,
                'completion_percentage': (len(self.results) / self.config.n_calls) * 100 if self.config.n_calls > 0 else 0
            },
            'interrupt_safe': True,  # Flag indicating this was saved safely
            'timestamp': datetime.now().isoformat(),
            'last_saved': current_time
        }
        
        # Write atomically to prevent corruption
        temp_file = progress_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            # Atomic move (works on most filesystems)
            temp_file.replace(progress_file)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save progress atomically: {e}")
            # Fallback to direct write
            try:
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
            except Exception as e2:
                print(f"❌ Critical: Could not save progress at all: {e2}")
                raise
        
        if self.config.verbose and len(self.results) % 20 == 0:
            print(f"💾 Progress saved: {len(self.results)} evaluations ({progress_data['runtime_info']['completion_percentage']:.1f}% complete)")

    def _setup_signal_handlers(self):
        """Setup signal handlers for clean exit on Ctrl-C."""
        def signal_handler(signum, frame):
            print(f"\n\n⚠️  INTERRUPT RECEIVED (Ctrl-C)")
            print(f"🛡️  Safely shutting down optimization...")
            
            # Save current progress
            try:
                self._save_progress()
                print(f"💾 Progress saved successfully!")
                print(f"   Evaluations completed: {len(self.results)}")
                if self.best_result:
                    print(f"   Best reward so far: {self.best_result.reward_analysis.composite_reward:.4f}")
                    print(f"   Best result saved to: {self.experiment_dir}")
                print(f"   Experiment directory: {self.experiment_dir}")
            except Exception as e:
                print(f"❌ Failed to save progress: {e}")
            
            # Cleanup resources
            try:
                if hasattr(self, 'performance_evaluator') and self.performance_evaluator:
                    if hasattr(self.performance_evaluator, 'cleanup'):
                        self.performance_evaluator.cleanup()
                        print(f"🧹 Cleaned up performance evaluator resources")
            except Exception as e:
                print(f"⚠️  Warning during cleanup: {e}")
            
            print(f"\n✅ Safe shutdown complete. You can resume optimization later!")
            print(f"💡 Tip: Use hybrid seeding to continue from where you left off")
            
            # Exit gracefully
            sys.exit(0)
        
        # Register the handler
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):  # Handle kill signals too
            signal.signal(signal.SIGTERM, signal_handler)

def run_clean_optimization_experiment(
    experiment_name: str = "clean_neural_optimization",
    n_calls: int = 50,
    database_paths: List[str] = None,
    sample_size: int = 5,
    comparisons_per_sample: int = 20,
    enable_performance_features: bool = True,
    hybrid_seed_path: Optional[str] = None,
    focus_parameters: bool = True,
    verbose: bool = True
) -> OptimizationResult:
    """
    Convenience function to run a complete clean optimization experiment.
    
    Args:
        experiment_name: Name for the experiment
        n_calls: Number of optimization evaluations
        database_paths: Paths to memory databases
        sample_size: Number of sample nodes per evaluation  
        comparisons_per_sample: Comparisons per sample
        enable_performance_features: Enable multi-fidelity and performance optimizations
        hybrid_seed_path: Path to hybrid seed configuration
        focus_parameters: Use focused parameter bounds (NOW ACTUALLY USED!)
        verbose: Enable verbose logging
        
    Returns:
        Best optimization result
    """
    
    # Load hybrid seed if provided
    hybrid_seed_config = None
    if hybrid_seed_path and Path(hybrid_seed_path).exists():
        with open(hybrid_seed_path, 'r') as f:
            hybrid_seed_config = json.load(f)
    
    if focus_parameters:
        parameter_bounds = CalculatorFactory.get_focused_parameter_bounds()
        if verbose:
            print("🎯 Using focused parameter bounds for high-impact parameters")
    else:
        parameter_bounds = CalculatorFactory.get_parameter_bounds()
        if verbose:
            print("📊 Using standard parameter bounds")
    
    config = OptimizationConfig(
        experiment_name=experiment_name,
        n_calls=n_calls,
        database_paths=database_paths,
        sample_size=sample_size,
        comparisons_per_sample=comparisons_per_sample,
        n_initial_points=min(10, n_calls // 5),
        parameter_bounds=parameter_bounds,
        hybrid_seed_config=hybrid_seed_config,
        verbose=verbose
    )
    
    performance_evaluator = None
    if enable_performance_features:
        from performance_optimizer import PerformanceOptimizedEvaluator, OptimizedConfig
        
        perf_config = OptimizedConfig(
            enable_multi_fidelity=True,
            enable_early_stopping=True,
            enable_parallel_evaluation=True,
            max_parallel_workers=2
        )
        
        performance_evaluator = PerformanceOptimizedEvaluator(perf_config, None)
    
    optimizer = CleanBayesianSemanticOptimizer(config, performance_evaluator, use_focused_bounds=focus_parameters)
    return optimizer.optimize()

def quick_clean_optimization_test():
    """Run a quick optimization test with clean architecture."""
    print("🧪 RUNNING CLEAN OPTIMIZATION TEST")
    print("=" * 50)
    
    # Quick test with minimal parameters
    result = run_clean_optimization_experiment(
        experiment_name="clean_test",
        n_calls=10,  # Very small for testing
        sample_size=3,
        comparisons_per_sample=10,
        enable_performance_features=True,
        focus_parameters=True,
        verbose=True
    )
    
    if result:
        print(f"\n✅ Clean optimization test completed successfully!")
        print(f"Best reward: {result.reward_analysis.composite_reward:.4f}")
        print(f"Component CVs: {result.reward_analysis.component_cvs}")
    else:
        print(f"\n❌ Clean optimization test failed!")
    
    return result


if __name__ == "__main__":
    # Run quick test when script is executed directly
    quick_clean_optimization_test()