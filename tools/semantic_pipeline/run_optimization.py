#!/usr/bin/env python3
"""
Clean Neuromorphic Memory Optimization Runner
Production-ready command-line interface for Bayesian parameter optimization.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import traceback

# Core optimization imports
from bayesian_optimizer import OptimizationConfig, CleanBayesianSemanticOptimizer
from performance_optimizer import (
    PerformanceOptimizedEvaluator, OptimizedConfig, 
    create_production_optimizer_config, load_hybrid_seed_config,
    estimate_optimization_speedup
)
from calc_integration import CalculatorFactory
from test_harness import TestConfiguration, SemanticTestHarness
from reward_functions import DiscriminationRewardFunction


class CleanOptimizationRunner:
    """
    Clean optimization runner with dependency injection and performance features.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.reward_function = DiscriminationRewardFunction()
        
    def log(self, message: str, level: str = "info"):
        """Clean logging instead of print patching."""
        if not self.verbose and level == "debug":
            return
            
        prefixes = {
            "info": "📋",
            "success": "✅", 
            "warning": "⚠️",
            "error": "❌",
            "debug": "🔧"
        }
        
        prefix = prefixes.get(level, "📋")
        print(f"{prefix} {message}")
    
    def create_base_evaluator(self, config: OptimizationConfig) -> callable:
        """
        Create base evaluation function with dependency injection.
        """
        
        def evaluate_parameter_vector(parameter_vector, sample_size: int, comparisons_per_sample: int) -> float:
            """
            Clean evaluation function with injected dependencies.
            """
            try:
                # Create calculator from parameter vector (dependency injection)
                calculator = CalculatorFactory.create_from_vector(
                    parameter_vector,
                    embedding_provider=None,  # Set by test harness
                    base_config_name="baseline"
                )
                
                # Create test configuration for this evaluation
                test_config = TestConfiguration(
                    sample_size=sample_size,
                    comparisons_per_sample=comparisons_per_sample,
                    database_paths=config.database_paths,
                    embedding_providers=config.embedding_providers,
                    calculator_configs=["custom"],  # Ignored - we inject directly
                    output_dir=str(Path(config.output_dir) / "temp_eval"),
                    generate_detailed_logs=False  # Skip logs for optimization
                )
                
                # Run evaluation with injected calculator
                harness = SemanticTestHarness(test_config)
                
                # CLEAN DEPENDENCY INJECTION: Pass calculator directly
                harness.injected_calculator = calculator
                
                experiment_results = harness.run_full_experiment()
                
                # Calculate reward
                reward_analysis = self.reward_function.composite_reward(
                    experiment_results,
                    parameter_vector,
                    CalculatorFactory.get_parameter_bounds()
                )
                
                return reward_analysis.composite_reward
                
            except Exception as e:
                self.log(f"Evaluation failed: {e}", "error")
                return 0.0
        
        return evaluate_parameter_vector
    
    def run_optimization(self, 
                        n_calls: int = 100,
                        database_paths: List[str] = None,
                        embedding_providers: List[str] = None,
                        output_dir: str = None,
                        hybrid_seed_path: str = None,
                        enable_optimizations: bool = True,
                        focus_parameters: bool = True,
                        max_parallel_workers: int = 2) -> Optional[str]:
        """
        Run complete optimization with clean architecture.
        """
        
        # Validate inputs
        if not database_paths:
            database_paths = self._find_databases()
            if not database_paths:
                self.log("No databases found", "error")
                return None
        
        if not embedding_providers:
            embedding_providers = ["stella"]
        
        if not output_dir:
            output_dir = f"optimization_results/clean_{int(time.time())}"
        
        # Load hybrid seed if provided
        hybrid_seed = None
        if hybrid_seed_path:
            hybrid_seed = load_hybrid_seed_config(hybrid_seed_path)
        
        # Create optimization configuration
        perf_config = create_production_optimizer_config(
            n_calls=n_calls,
            enable_all_optimizations=enable_optimizations,
            focus_parameters=focus_parameters
        )
        
        config = OptimizationConfig(
            experiment_name="clean_optimization",
            n_calls=n_calls,
            n_initial_points=perf_config['n_initial_points'],
            sample_size=perf_config['sample_size'],
            comparisons_per_sample=perf_config['comparisons_per_sample'],
            database_paths=database_paths,
            embedding_providers=embedding_providers,
            output_dir=output_dir,
            parameter_bounds=perf_config.get('parameter_bounds'),
            hybrid_seed_config=hybrid_seed,
            verbose=self.verbose
        )
        
        self.log(f"🚀 STARTING OPTIMIZATION")
        self.log(f"   Databases: {len(database_paths)}")
        self.log(f"   Providers: {embedding_providers}")
        self.log(f"   Evaluations: {n_calls}")
        self.log(f"   Optimizations: {'enabled' if enable_optimizations else 'disabled'}")
        self.log(f"   Focused params: {'yes' if focus_parameters else 'no'}")
        if hybrid_seed:
            self.log(f"   Hybrid seed: {hybrid_seed_path}")
        
        try:
            # Create base evaluator function
            base_evaluator = self.create_base_evaluator(config)
            
            # Create optimized configuration for performance features
            optimized_config = OptimizedConfig(
                enable_multi_fidelity=enable_optimizations,
                enable_early_stopping=enable_optimizations,
                enable_parallel_evaluation=enable_optimizations,
                max_parallel_workers=max_parallel_workers,
                enable_surrogate_modeling=enable_optimizations,
                focus_embedding_independent=focus_parameters
            )
            
            # Create performance-optimized evaluator
            performance_evaluator = PerformanceOptimizedEvaluator(optimized_config, base_evaluator)
            
            # Create clean optimizer with dependency injection
            optimizer = CleanBayesianSemanticOptimizer(config, performance_evaluator)

            
            # Run optimization
            start_time = time.time()
            best_result = optimizer.optimize()
            total_time = time.time() - start_time
            
            if best_result:
                self.log(f"🎉 OPTIMIZATION COMPLETE", "success")
                self.log(f"   Runtime: {total_time/60:.1f} minutes")
                self.log(f"   Best reward: {best_result.reward_analysis.composite_reward:.4f}")
                self.log(f"   Results: {optimizer.experiment_dir}")
                
                # Show component CVs
                if hasattr(best_result.reward_analysis, 'component_cvs'):
                    self.log("   Component CVs:")
                    for comp, cv in best_result.reward_analysis.component_cvs.items():
                        status = "🟢" if cv >= 0.6 else "🟡" if cv >= 0.4 else "🟠" if cv >= 0.2 else "🔴"
                        self.log(f"     {comp:20} {cv:.3f} {status}")
                
                return str(optimizer.experiment_dir)
            else:
                self.log("Optimization failed", "error")
                return None
                
        except Exception as e:
            self.log(f"Optimization error: {e}", "error")
            if self.verbose:
                traceback.print_exc()
            return None
    
    def _find_databases(self) -> List[str]:
        """Find available database files."""
        db_dir = Path("dbs")
        if not db_dir.exists():
            return []
        
        available_dbs = list(db_dir.glob("*.db"))
        return [str(db) for db in available_dbs]


def main():
    """Clean command-line interface."""
    
    parser = argparse.ArgumentParser(
        description="Clean Neuromorphic Memory Optimization Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard optimization with all features
  python run_optimization.py optimize --evaluations 100 --providers stella

  # Fast optimization with focused parameters  
  python run_optimization.py optimize --evaluations 50 --focus-parameters --max-workers 3

  # Optimization with hybrid seed
  python run_optimization.py optimize --hybrid-seed hybrid_config.json --evaluations 75"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Run optimization')
    opt_parser.add_argument('--evaluations', type=int, default=100, help='Number of evaluations')
    opt_parser.add_argument('--databases', nargs='+', help='Database paths (auto-detect if not provided)')
    opt_parser.add_argument('--providers', nargs='+', default=['stella'], help='Embedding providers')
    opt_parser.add_argument('--output-dir', help='Output directory (auto-generated if not provided)')
    opt_parser.add_argument('--hybrid-seed', help='Path to hybrid seed configuration')
    opt_parser.add_argument('--no-optimizations', action='store_true', help='Disable performance optimizations')
    opt_parser.add_argument('--no-focus-parameters', action='store_true', help='Use full parameter space')
    opt_parser.add_argument('--max-workers', type=int, default=2, help='Maximum parallel workers')
    opt_parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    if args.command == 'optimize':
        runner = CleanOptimizationRunner(verbose=not args.quiet)
        
        result_dir = runner.run_optimization(
            n_calls=args.evaluations,
            database_paths=args.databases,
            embedding_providers=args.providers,
            output_dir=args.output_dir,
            hybrid_seed_path=args.hybrid_seed,
            enable_optimizations=not args.no_optimizations,
            focus_parameters=not args.no_focus_parameters,
            max_parallel_workers=args.max_workers
        )
        
        if result_dir:
            print(f"\n✅ Optimization complete: {result_dir}")
        else:
            print(f"\n❌ Optimization failed")
            sys.exit(1)
        
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# Convenience function for compatibility
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
) -> Optional[str]:
    """Convenience function for clean optimization experiment."""
    
    runner = CleanOptimizationRunner(verbose=verbose)
    return runner.run_optimization(
        n_calls=n_calls,
        database_paths=database_paths,
        embedding_providers=["stella"],
        enable_optimizations=enable_performance_features,
        focus_parameters=focus_parameters,
        hybrid_seed_path=hybrid_seed_path
    )