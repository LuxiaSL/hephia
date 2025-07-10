#!/usr/bin/env python3
"""
run_semantic_tests.py

CLI script for running semantic analysis experiments.
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Check for progress bar support
try:
    import tqdm
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False

# Check for parameterized calculator availability
try:
    from calc_integration import CalculatorFactory
    PARAMETERIZED_AVAILABLE = True
except ImportError:
    PARAMETERIZED_AVAILABLE = False

try:
    from test_harness import TestConfiguration, SemanticTestHarness, run_quick_experiment
    from embedding_providers import COMMON_MODELS
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure semantic_calculator.py, embedding_providers.py, and test_harness.py are in the same directory")
    sys.exit(1)


def _get_config_description(config_name: str) -> str:
    """Get human-readable description of calculator configuration."""
    descriptions = {
        "baseline": "Original implementation (25% cohesion, 25% NE density)",
        "density_focused": "Boost NE density to 35% (entity-heavy content)",
        "cohesion_focused": "Boost semantic cohesion to 40% (narrative content)",
        "balanced": "Equal 20% distribution across components",
        "complexity_focused": "Boost logical complexity to 35% (reasoning content)",
        "gentle_transform": "Softer discrimination curves (less aggressive)",
        "aggressive_transform": "Steeper discrimination curves (more aggressive)",
        "fast_heuristic": "Performance-optimized (word overlap, heuristics)",
        "slow_comprehensive": "Analysis-optimized (full embedding similarity)",
        "optimized": "Performance-balanced configuration",
        "comprehensive": "Maximum analytical depth"
    }
    return descriptions.get(config_name, "Custom configuration")


def main():
    parser = argparse.ArgumentParser(
        description="Run semantic analysis experiments on memory databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 2 databases and default models
  python run_semantic_tests.py quick /path/to/db1.db /path/to/db2.db

  # Test different calculator configurations
  python run_semantic_tests.py quick /path/to/db1.db /path/to/db2.db \\
    --calculators baseline density_focused cohesion_focused

  # Full experiment with custom parameters
  python run_semantic_tests.py full /path/to/db1.db /path/to/db2.db /path/to/db3.db \\
    --sample-size 100 --comparisons 200 --providers stella all_mpnet bge_large \\
    --calculators baseline balanced aggressive_transform

  # Use custom configuration file (parameterized system)
  python run_semantic_tests.py quick /path/to/db1.db /path/to/db2.db \\
    --calculators custom:my_optimized_config.json

  # List available models and configurations
  python run_semantic_tests.py list-models
  python run_semantic_tests.py list-calculators
        """)
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Quick experiment command
    quick_parser = subparsers.add_parser('quick', help='Run quick experiment with defaults')
    quick_parser.add_argument('databases', nargs='+', help='Database file paths')
    quick_parser.add_argument('--sample-size', type=int, default=20, help='Sample size per database (default: 20)')
    quick_parser.add_argument('--comparisons', type=int, default=50, help='Comparisons per sample (default: 50)')
    quick_parser.add_argument('--providers', nargs='+', default=['stella', 'all_mpnet'], 
                             help='Embedding providers to test (default: stella all_mpnet)')
    quick_parser.add_argument('--calculators', nargs='+', default=['baseline'], 
                             help='Calculator configurations to test (default: baseline)')
    quick_parser.add_argument('--output', default=None, help='Output directory (default: auto-generated)')
    
    # Full experiment command  
    full_parser = subparsers.add_parser('full', help='Run full experiment with custom configuration')
    full_parser.add_argument('databases', nargs='+', help='Database file paths')
    full_parser.add_argument('--sample-size', type=int, default=50, help='Sample size per database (default: 50)')
    full_parser.add_argument('--comparisons', type=int, default=100, help='Comparisons per sample (default: 100)')
    full_parser.add_argument('--providers', nargs='+', default=['stella', 'all_mpnet'], 
                             help='Embedding providers to test')
    full_parser.add_argument('--calculators', nargs='+', default=['baseline'],
                             help='Calculator configurations to test (default: baseline)')
    full_parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    full_parser.add_argument('--output', required=True, help='Output directory')
    full_parser.add_argument('--no-logs', action='store_true', help='Skip generating detailed log files')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available embedding models')
    
    # List calculators command
    calc_parser = subparsers.add_parser('list-calculators', help='List available calculator configurations')
    
    args = parser.parse_args()
    
    if args.command == 'list-models':
        print("Available embedding models:")
        print("\nBuilt-in aliases:")
        for alias, model_id in COMMON_MODELS.items():
            print(f"  {alias:12} -> {model_id}")
        print("\nSpecial providers:")
        print("  stella       -> StellaEmbeddingProvider (your current production model)")
        print("  openai       -> OpenAIEmbeddingProvider (requires API access)")
        print("  test         -> DeterministicTestProvider (for debugging)")
        print("\nYou can also use any HuggingFace model ID directly")
        return
    
    if args.command == 'list-calculators':
        print("Available calculator configurations:")
        
        if PARAMETERIZED_AVAILABLE:
            print("\n🤖 Parameterized Configurations (Neural Optimization Ready):")
            try:
                configs = CalculatorFactory.get_available_configs()
                for config in sorted(configs):
                    if config == "current":
                        continue
                    print(f"  {config:20} -> {_get_config_description(config)}")
            except Exception as e:
                print(f"  Error loading parameterized configs: {e}")
        else:
            print("\n⚠️  Parameterized calculator system not available")
            print("   Install: Make sure parameterized_calculator.py and calculator_integration.py are present")
        
        print("\n📚 Legacy Configurations:")
        print("  current              -> Original semantic calculator (baseline)")
        
        print("\n💡 Usage examples:")
        print("  python run_semantic_tests.py quick ./dbs/*.db --calculators baseline")
        print("  python run_semantic_tests.py full ./dbs/*.db --calculators baseline density_focused cohesion_focused")
        
        if PARAMETERIZED_AVAILABLE:
            print("  python run_semantic_tests.py quick ./dbs/*.db --calculators custom:my_config.json")
        
        return
    
    if not args.command:
        parser.print_help()
        return
    
    # Validate database files exist
    for db_path in args.databases:
        if not Path(db_path).exists():
            print(f"Error: Database file not found: {db_path}")
            sys.exit(1)
    
    print(f"Found {len(args.databases)} database(s)")
    for db_path in args.databases:
        print(f"  - {db_path}")
    
    # Progress bar status
    if PROGRESS_AVAILABLE:
        print("✓ Progress bars enabled")
    else:
        print("⚠ No progress bars (install tqdm: pip install tqdm)")
    
    # Calculator system status
    if PARAMETERIZED_AVAILABLE:
        print("✓ Parameterized calculator system available (neural optimization ready)")
    else:
        print("⚠ Parameterized calculator system not available (legacy mode)")
    
    if args.command == 'quick':
        print(f"\nRunning quick experiment:")
        print(f"  Sample size: {args.sample_size} nodes per database")
        print(f"  Comparisons: {args.comparisons} per sample node")
        print(f"  Providers: {', '.join(args.providers)}")
        
        try:
            results = run_quick_experiment(
                database_paths=args.databases,
                sample_size=args.sample_size,
                comparisons_per_sample=args.comparisons,
                providers=args.providers
            )
            
            # Print summary
            print("\n" + "="*60)
            print("QUICK EXPERIMENT SUMMARY")
            print("="*60)
            summary = results['analysis_summary']
            timing = results['timing']
            
            print(f"Total comparisons: {summary.get('total_comparisons', 0):,}")
            print(f"Duration: {timing['total_duration']:.1f}s")
            print(f"Rate: {timing['comparisons_per_second']:.1f} comparisons/sec")
            
            # Show cache efficiency
            if summary.get('total_comparisons', 0) > 0:
                print(f"Cache efficiency: Processed only sampled nodes (not entire databases)")
            
            # Only show detailed analysis if we have successful results
            if summary.get('total_comparisons', 0) > 0 and 'error' not in summary:
                if 'embedding_similarity_analysis' in summary and 'error' not in summary['embedding_similarity_analysis']:
                    sim_analysis = summary['embedding_similarity_analysis']
                    print(f"Embedding similarity: mean={sim_analysis['mean']:.3f}, cv={sim_analysis['coefficient_of_variation']:.3f}")
                
                if 'semantic_density_analysis' in summary and 'error' not in summary['semantic_density_analysis']:
                    density_analysis = summary['semantic_density_analysis']['distribution']
                    print(f"Semantic density: mean={density_analysis['mean']:.3f}, cv={density_analysis['coefficient_of_variation']:.3f}")
                
                if 'provider_comparison' in summary and 'error' not in summary['provider_comparison']:
                    print("\nProvider comparison:")
                    for provider, data in summary['provider_comparison'].items():
                        emb_cv = data['embedding_similarity']['coefficient_of_variation']
                        print(f"  {provider}: embedding_cv={emb_cv:.3f}")
            else:
                if 'error' in summary:
                    print(f"⚠️  {summary['error']}")
                else:
                    print("⚠️  No successful comparisons completed")
            
            print(f"\nDetailed results saved to output directory")
            
        except Exception as e:
            print(f"Error running experiment: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.command == 'full':
        print(f"\nRunning full experiment:")
        print(f"  Sample size: {args.sample_size} nodes per database")
        print(f"  Comparisons: {args.comparisons} per sample node")
        print(f"  Providers: {', '.join(args.providers)}")
        print(f"  Calculators: {', '.join(args.calculators)}")
        print(f"  Output: {args.output}")
        
        config = TestConfiguration(
            sample_size=args.sample_size,
            comparisons_per_sample=args.comparisons,
            database_paths=args.databases,
            embedding_providers=args.providers,
            calculator_configs=args.calculators,
            random_seed=args.seed,
            output_dir=args.output,
            generate_detailed_logs=not args.no_logs
        )
        
        try:
            harness = SemanticTestHarness(config)
            results = harness.run_full_experiment()
            
            # Print summary
            print("\n" + "="*60)
            print("FULL EXPERIMENT SUMMARY")
            print("="*60)
            
            timing = results['timing']
            print(f"Duration: {timing['total_duration']:.1f}s")
            print(f"Rate: {timing['comparisons_per_second']:.1f} comparisons/sec")
            
            summary = results['analysis_summary']
            print(f"Total comparisons: {summary['total_comparisons']:,}")
            
            if 'provider_comparison' in summary:
                print("\nProvider discrimination comparison:")
                for provider, data in summary['provider_comparison'].items():
                    emb_cv = data['embedding_similarity']['coefficient_of_variation']
                    density_cv = data['semantic_density']['coefficient_of_variation']
                    print(f"  {provider}: emb_cv={emb_cv:.3f}, density_cv={density_cv:.3f}")
            
            if 'component_discrimination_analysis' in summary:
                print("\nComponent discrimination analysis:")
                for comp, data in summary['component_discrimination_analysis'].items():
                    cv = data['coefficient_of_variation']
                    print(f"  {comp}: cv={cv:.3f}")
            
            print(f"\nDetailed results saved to: {args.output}")
            
        except Exception as e:
            print(f"Error running experiment: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()