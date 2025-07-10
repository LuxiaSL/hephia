#!/usr/bin/env python3
"""
Optimized Neuromorphic Memory Optimization Runner
- Fixes model reloading (caches embedding providers globally)
- Suppresses all progress bars and verbose output
- Includes results analysis script

Usage:
    python run_optimization.py run quick --provider stella
    python run_optimization.py analyze results_directory/
"""

import argparse
import time
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import glob

def suppress_all_output():
    """Suppress all verbose output and progress bars."""
    
    # Suppress tqdm globally
    os.environ['TQDM_DISABLE'] = '1'
    
    # Suppress sentence-transformers logging
    import logging
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Suppress other noisy loggers
    for logger_name in ['urllib3', 'requests', 'PIL', 'matplotlib']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def fix_embedding_provider_caching():
    """Fix embedding providers to use global caching."""
    
    print("🔧 Fixing embedding provider caching...")
    
    # Create a global provider cache module
    cache_code = '''
"""
Global embedding provider cache to prevent model reloading.
"""

_GLOBAL_PROVIDER_CACHE = {}

def get_cached_provider(provider_alias: str):
    """Get cached provider or create if not exists."""
    if provider_alias not in _GLOBAL_PROVIDER_CACHE:
        from embedding_providers import create_provider_by_alias
        print(f"🔄 Loading {provider_alias} embedding model (first time only)...")
        _GLOBAL_PROVIDER_CACHE[provider_alias] = create_provider_by_alias(provider_alias)
        print(f"✅ {provider_alias} loaded and cached")
    return _GLOBAL_PROVIDER_CACHE[provider_alias]

def clear_cache():
    """Clear the global cache."""
    global _GLOBAL_PROVIDER_CACHE
    _GLOBAL_PROVIDER_CACHE.clear()
'''
    
    with open('embedding_cache.py', 'w') as f:
        f.write(cache_code)
    
    # Fix test_harness.py to use cached providers
    with open('test_harness.py', 'r') as f:
        content = f.read()
    
    # Replace the provider creation
    old_create = '''    def _create_embedding_provider(self, provider_alias: str) -> EmbeddingProvider:
        """Create embedding provider by alias (cached version)."""
        if provider_alias in self._provider_cache:
            return self._provider_cache[provider_alias]
        
        # Import here to avoid circular dependencies
        from embedding_providers import create_provider_by_alias
        provider = create_provider_by_alias(provider_alias)
        self._provider_cache[provider_alias] = provider
        return provider'''
    
    new_create = '''    def _create_embedding_provider(self, provider_alias: str) -> EmbeddingProvider:
        """Create embedding provider using global cache."""
        from embedding_cache import get_cached_provider
        return get_cached_provider(provider_alias)'''
    
    if old_create in content:
        content = content.replace(old_create, new_create)
    else:
        # Fallback approach - replace any provider creation
        content = content.replace(
            'from embedding_providers import create_provider_by_alias\n        provider = create_provider_by_alias(provider_alias)',
            'from embedding_cache import get_cached_provider\n        provider = get_cached_provider(provider_alias)'
        )
    
    # Disable progress bars in test harness
    content = content.replace(
        'progress_bar(iterable, desc="Processing", disable=False)',
        'progress_bar(iterable, desc="Processing", disable=True)'
    )
    content = content.replace(
        'disable=False',
        'disable=True'
    )
    
    with open('test_harness.py', 'w') as f:
        f.write(content)
    
    print("  ✅ Fixed embedding provider caching")
    print("  ✅ Disabled all progress bars")

def clean_optimizer_output():
    """Clean up optimizer output for minimal logging."""
    
    with open('bayesian_optimizer.py', 'r') as f:
        content = f.read()
    
    # Fix 1: Custom output directory handling
    old_experiment_dir = '''        # Create experiment directory
        self.experiment_dir = Path(self.config.output_dir) / f"{self.config.experiment_name}_{int(time.time())}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)'''
    
    new_experiment_dir = '''        # Create experiment directory - respect custom output dirs
        if hasattr(self.config, '_custom_output_dir') and self.config._custom_output_dir:
            # Use exact custom directory specified by user
            self.experiment_dir = Path(self.config.output_dir)
        else:
            # Use auto-generated directory structure
            self.experiment_dir = Path(self.config.output_dir) / f"{self.config.experiment_name}_{int(time.time())}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)'''
    
    if old_experiment_dir in content:
        content = content.replace(old_experiment_dir, new_experiment_dir)
    
    # Fix 2: Replace verbose evaluation output
    content = content.replace(
        '''            if evaluation_id % 5 == 1:  # Every 5th evaluation
                print(f"🔬 Evaluations {evaluation_id}-{min(evaluation_id+4, self.config.n_calls)}/{self.config.n_calls}")''',
        '''            if evaluation_id == 1 or evaluation_id % 10 == 0:  # Less frequent updates
                print(f"🔬 Evaluation {evaluation_id}/{self.config.n_calls}")'''
    )
    
    # Fix 3: Simplify best result reporting
    content = content.replace(
        '''            if reward_analysis.composite_reward > (self.best_result.reward_analysis.composite_reward if self.best_result else 0):
                print(f"🏆 NEW BEST: {reward_analysis.composite_reward:.4f} (CVs: {reward_analysis.component_cvs})")
            elif evaluation_id % 5 == 0:  # Progress update every 5 evaluations
                print(f"   Eval {evaluation_id}: reward={reward_analysis.composite_reward:.4f}, best={self.best_result.reward_analysis.composite_reward:.4f}")''',
        '''            if reward_analysis.composite_reward > (self.best_result.reward_analysis.composite_reward if self.best_result else 0):
                print(f"🏆 NEW BEST: {reward_analysis.composite_reward:.4f}")
                for comp, cv in reward_analysis.component_cvs.items():
                    print(f"     {comp}: {cv:.3f}")'''
    )
    
    # Fix 4: Database path handling - use all available databases
    content = content.replace(
        'self.database_paths = existing_paths[:1] if existing_paths else ["test_memories.db"]',
        'self.database_paths = existing_paths if existing_paths else ["dbs/cypher.db", "dbs/haiku.db", "dbs/sonnet.db"]'
    )
    
    with open('bayesian_optimizer.py', 'w') as f:
        f.write(content)
    
    print("  ✅ Cleaned optimizer output and fixed custom output directories")


def extract_summary_from_progress(progress: dict) -> dict:
    """Extract summary information from optimization_progress.json."""
    
    if not progress.get('results'):
        return {'error': 'No results found in progress file'}
    
    # Find best result - FIX: Use correct path for composite reward
    try:
        best_result = max(progress['results'], key=lambda x: x['reward_analysis']['scores']['composite'])
    except KeyError as e:
        # Fallback: handle different possible structures
        print(f"⚠️  Warning: Expected structure not found, trying fallback... ({e})")
        try:
            best_result = max(progress['results'], key=lambda x: x['reward_analysis']['composite_reward'])
        except KeyError:
            return {'error': 'Unable to find composite reward in results structure'}
    
    # Calculate success rate
    successful_evals = sum(1 for r in progress['results'] 
                          if r['reward_analysis']['scores']['composite'] > 0)
    success_rate = successful_evals / len(progress['results']) * 100
    
    # Extract provider info (from first result)
    first_result = progress['results'][0]
    provider = first_result.get('experiment_metadata', {}).get('test_config', {}).get('embedding_providers', ['unknown'])[0]
    
    # Calculate runtime (from timestamps)
    if len(progress['results']) > 1:
        start_time = progress['results'][0]['timestamp']
        end_time = progress['results'][-1]['timestamp']
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        runtime_minutes = (end_dt - start_dt).total_seconds() / 60
    else:
        runtime_minutes = 0.0
    
    # Extract component CVs - FIX: Use correct path with fallback
    try:
        best_component_cvs = best_result['reward_analysis']['details']['component_cvs']
    except KeyError:
        # Fallback: try experiment metadata
        best_component_cvs = best_result.get('experiment_metadata', {}).get('component_cvs', {})
    
    # Calculate comparisons per second
    total_comparisons = sum(r.get('experiment_metadata', {}).get('total_comparisons', 0) for r in progress['results'])
    comparisons_per_second = total_comparisons / (runtime_minutes * 60) if runtime_minutes > 0 else 0
    
    return {
        'experiment_type': 'neuromorphic_optimization',
        'embedding_provider': provider,
        'total_evaluations': len(progress['results']),
        'successful_evaluations': successful_evals,
        'success_rate': success_rate,
        'best_reward': best_result['reward_analysis']['scores']['composite'],
        'best_component_cvs': best_component_cvs,
        'runtime_minutes': runtime_minutes,
        'comparisons_per_second': comparisons_per_second,
        'configuration': {
            'n_calls': progress.get('config', {}).get('n_calls', 0),
            'parameter_space': len(progress.get('parameter_names', [])),
            'best_evaluation_id': best_result['evaluation_id']
        }
    }

def analyze_optimization_results(results_dir: str):
    """Analyze optimization results and provide insights."""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print(f"📊 ANALYZING OPTIMIZATION RESULTS")
    print(f"Directory: {results_path}")
    print("=" * 60)
    
    # Try to find summary file first
    summary_files = list(results_path.glob("**/real_optimization_summary.json"))
    if not summary_files:
        summary_files = list(results_path.glob("**/optimization_summary.json"))
    
    if summary_files:
        # Use summary file if available
        summary_file = summary_files[0]
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"📋 Using summary file: {summary_file.name}")
    else:
        # Fallback: create summary from optimization_progress.json
        progress_files = list(results_path.glob("**/optimization_progress.json"))
        if not progress_files:
            print("❌ No summary or progress file found")
            return
        
        progress_file = progress_files[0]
        print(f"📋 Using progress file: {progress_file.name}")
        
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        # Extract summary from progress file
        summary = extract_summary_from_progress(progress)
        
        # Handle extraction errors
        if 'error' in summary:
            print(f"❌ {summary['error']}")
            return
    
    # Basic stats
    print(f"🎯 OPTIMIZATION SUMMARY")
    print(f"   Embedding provider: {summary.get('embedding_provider', 'unknown')}")
    print(f"   Total evaluations: {summary['total_evaluations']}")
    print(f"   Successful evaluations: {summary.get('successful_evaluations', '?')}")
    print(f"   Success rate: {summary.get('success_rate', 0):.1f}%")
    print(f"   Runtime: {summary['runtime_minutes']:.1f} minutes")
    print(f"   Rate: {summary.get('comparisons_per_second', 0):.1f} comparisons/sec")
    print()
    
    # Best results
    print(f"🏆 BEST CONFIGURATION FOUND")
    print(f"   Composite reward: {summary['best_reward']:.4f}")
    print(f"   Component discrimination (CV scores):")
    
    best_cvs = summary['best_component_cvs']
    for comp_name, cv_value in best_cvs.items():
        if cv_value >= 0.6:
            status = "🟢 EXCELLENT"
        elif cv_value >= 0.4:
            status = "🟡 GOOD"
        elif cv_value >= 0.2:
            status = "🟠 WEAK"
        else:
            status = "🔴 CLUSTERING"
        
        print(f"     {comp_name:20} CV={cv_value:.3f} {status}")
    
    print()
    
    # Performance analysis
    print(f"📈 PERFORMANCE ANALYSIS")
    
    # Count components by performance tier
    excellent = sum(1 for cv in best_cvs.values() if cv >= 0.6)
    good = sum(1 for cv in best_cvs.values() if 0.4 <= cv < 0.6)
    weak = sum(1 for cv in best_cvs.values() if 0.2 <= cv < 0.4)
    clustering = sum(1 for cv in best_cvs.values() if cv < 0.2)
    
    print(f"   🟢 Excellent discriminators (CV≥0.6): {excellent}/6")
    print(f"   🟡 Good discriminators (CV≥0.4): {good}/6")
    print(f"   🟠 Weak discriminators (CV≥0.2): {weak}/6")
    print(f"   🔴 Clustering discriminators (CV<0.2): {clustering}/6")
    
    total_discrimination = sum(best_cvs.values())
    print(f"   📊 Total discrimination power: {total_discrimination:.3f}/6.0")
    print(f"   📊 Average CV: {total_discrimination/6:.3f}")
    
    print()
    
    # Improvement recommendations
    print(f"🔧 IMPROVEMENT RECOMMENDATIONS")
    
    problem_components = [(name, cv) for name, cv in best_cvs.items() if cv < 0.3]
    if problem_components:
        print(f"   Priority fixes needed:")
        for comp_name, cv_value in problem_components:
            if cv_value < 0.1:
                issue = "severe clustering - check amplification parameters"
            elif cv_value < 0.2:
                issue = "mild clustering - adjust normalization factors"
            else:
                issue = "low discrimination - tune component weights"
            print(f"     {comp_name} (CV={cv_value:.3f}): {issue}")
    else:
        print(f"   ✅ All components functional - focus on fine-tuning")
    
    # Find best parameters if available
    progress_files = list(results_path.glob("**/optimization_progress.json"))
    if progress_files:
        print(f"\n📋 BEST PARAMETER CONFIGURATION")
        
        with open(progress_files[0], 'r') as f:
            progress = json.load(f)
        
        if progress['results']:
            # FIX: Use correct path for composite reward
            best_eval = max(progress['results'], key=lambda x: x['reward_analysis']['scores']['composite'])
            best_params = best_eval['parameter_vector']
            param_names = progress['parameter_names']
            
            print(f"   From evaluation {best_eval['evaluation_id']}:")
            print(f"   Component weights:")
            
            # Show first 6 parameters (component weights)
            weight_names = ['topic_surprise', 'ne_density', 'conceptual_surprise', 
                          'logical_complexity', 'conceptual_bridging', 'information_density']
            for i, (name, weight) in enumerate(zip(weight_names, best_params[:6])):
                print(f"     {name:20}: {weight:.3f}")
            
            print(f"   (Full parameter vector has {len(best_params)} values)")
    
    print()
    print(f"📁 Full results available in: {results_path}")

def run_optimized_optimization(config_name: str, n_calls: int, sample_size: int, 
                             comparisons_per_sample: int, embedding_provider: str,
                             output_dir: str = None, experiment_name: str = None,
                             custom_output_dir: bool = False):
    """Run optimization with all fixes applied."""
    
    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = f"optimized_neuromorphic_{config_name}"
    
    # Generate output directory if not provided - always under optimization_results/
    if output_dir is None:
        output_dir = f"optimization_results/{config_name}_{int(time.time())}"
    elif not output_dir.startswith("optimization_results"):
        # Ensure it's under optimization_results/
        output_dir = f"optimization_results/{output_dir}"
    
    print(f"🧠 OPTIMIZED NEUROMORPHIC MEMORY OPTIMIZATION")
    print(f"=" * 50)
    print(f"Config: {config_name} | Provider: {embedding_provider}")
    print(f"Evaluations: {n_calls} | Samples: {sample_size} | Comparisons: {comparisons_per_sample}")
    print(f"Output: {output_dir}")
    print(f"🔇 Verbose output suppressed for clean progress")
    print(f"=" * 50)
    
    # Suppress all verbose output
    suppress_all_output()
    
    # Import after fixes
    from bayesian_optimizer import OptimizationConfig, BayesianSemanticOptimizer
    
    # Find databases
    db_dir = Path("dbs")
    available_dbs = list(db_dir.glob("*.db"))
    
    if not available_dbs:
        print("❌ No databases found in dbs/ directory")
        return None
    
    print(f"📊 Using {len(available_dbs)} databases: {[db.name for db in available_dbs]}")
    
    config = OptimizationConfig(
        experiment_name=experiment_name,
        n_calls=n_calls,
        n_initial_points=max(3, n_calls // 8),
        sample_size=sample_size,
        comparisons_per_sample=comparisons_per_sample,
        database_paths=[str(db) for db in available_dbs],  # Use ALL databases
        embedding_providers=[embedding_provider],
        output_dir=output_dir
    )
    
    # Mark if this is a custom output directory (so bayesian_optimizer respects it)
    config._custom_output_dir = custom_output_dir
    
    total_comparisons = n_calls * sample_size * comparisons_per_sample * len(available_dbs)
    estimated_minutes = total_comparisons / 50
    
    print(f"🚀 Starting optimization...")
    print(f"   Total comparisons: {total_comparisons:,}")
    print(f"   Estimated time: ~{estimated_minutes:.1f} minutes")
    print(f"   Model caching: ENABLED (no reloading)")
    print()
    
    try:
        optimizer = BayesianSemanticOptimizer(config)
        
        start_time = time.time()
        best_result = optimizer.optimize()
        total_time = time.time() - start_time
        
        if best_result:
            print(f"\n🎉 OPTIMIZATION COMPLETE!")
            print(f"   Runtime: {total_time/60:.1f} minutes")
            print(f"   Best reward: {best_result.reward_analysis.composite_reward:.4f}")
            print(f"   Success rate: {sum(1 for r in optimizer.results if r.reward_analysis.composite_reward > 0) / len(optimizer.results) * 100:.1f}%")
            
            # Auto-analyze results
            print(f"\n📊 AUTO-ANALYZING RESULTS...")
            analyze_optimization_results(str(optimizer.experiment_dir))
            
            return str(optimizer.experiment_dir)
        else:
            print(f"\n❌ Optimization failed")
            return None
            
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Optimized Neuromorphic Memory Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick presets
  python run_optimization.py run quick --provider stella
  python run_optimization.py run deep --provider all_minilm
  
  # Custom configurations
  python run_optimization.py run custom --n-calls 200 --sample-size 12 --comparisons 25
  python run_optimization.py run deep --n-calls 500 --output-dir "server_deep_run"
  
  # Server deep run example
  python run_optimization.py run custom --n-calls 1000 --sample-size 15 --comparisons 30 \\
    --provider stella --output-dir "server_optimization_run_1" --name "server_deep_optimization"
  
  # Analyze results
  python run_optimization.py analyze path/to/results/directory/
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Run optimization
    run_parser = subparsers.add_parser('run', help='Run optimization')
    run_parser.add_argument('config', choices=['quick', 'real', 'deep', 'custom'], 
                           help='Preset config or custom for full control')
    run_parser.add_argument('--provider', default='stella', choices=['stella', 'all_minilm', 'all_mpnet'],
                           help='Embedding provider to use')
    
    # Custom configuration arguments
    run_parser.add_argument('--n-calls', type=int, 
                           help='Number of optimization evaluations (overrides preset)')
    run_parser.add_argument('--sample-size', type=int,
                           help='Number of sample nodes per evaluation (overrides preset)')
    run_parser.add_argument('--comparisons', type=int,
                           help='Number of comparisons per sample (overrides preset)')
    run_parser.add_argument('--output-dir', type=str,
                           help='Custom output directory (default: auto-generated)')
    run_parser.add_argument('--name', type=str,
                           help='Custom experiment name (default: based on config)')
    
    # Analyze results
    analyze_parser = subparsers.add_parser('analyze', help='Analyze optimization results')
    analyze_parser.add_argument('results_dir', help='Path to results directory')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        # Apply all fixes first
        fix_embedding_provider_caching()
        clean_optimizer_output()
        
        presets = {
            'quick': {'n_calls': 25, 'sample_size': 6, 'comparisons': 12},
            'real': {'n_calls': 50, 'sample_size': 8, 'comparisons': 15},
            'deep': {'n_calls': 100, 'sample_size': 10, 'comparisons': 20}
        }
        
        config_params = presets[args.config]
        
        results_dir = run_optimized_optimization(
            config_name=args.config,
            n_calls=config_params['n_calls'],
            sample_size=config_params['sample_size'],
            comparisons_per_sample=config_params['comparisons'],
            embedding_provider=args.provider
        )
        
        if results_dir:
            print(f"\n✅ Results saved to: {results_dir}")
        
    elif args.command == 'analyze':
        analyze_optimization_results(args.results_dir)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()