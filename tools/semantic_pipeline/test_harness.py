"""
testing/test_harness.py

Test harness for systematic memory semantic analysis.
Samples nodes from existing databases and generates comparison metrics
identical to what collect_*.py scripts analyze.

Updated to support parameterized calculators for neural optimization.
"""

import asyncio
import json
import random
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import statistics
import numpy as np

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available - no progress bars. Install with: pip install tqdm")


from calc_integration import CalculatorFactory, create_calculator_for_testing
from semantic_calculator import ParameterizedSemanticCalculator
PARAMETERIZED_CALCULATOR_AVAILABLE = True

from embedding_providers import EmbeddingProvider


def progress_bar(iterable, desc="Processing", disable=False):
    """Create progress bar if tqdm available, otherwise return plain iterable."""
    if TQDM_AVAILABLE and not disable:
        return tqdm(iterable, desc=desc, unit="items")
    else:
        return iterable


@dataclass
class MemoryNode:
    """Represents a memory node from the database."""
    id: int
    timestamp: float
    text_content: str
    embedding: List[float]
    raw_state: str
    processed_state: str
    strength: float
    source_type: Optional[str] = None
    
    @classmethod
    def from_db_row(cls, row: Tuple) -> 'MemoryNode':
        """Create MemoryNode from database row."""
        return cls(
            id=row[0],
            timestamp=row[1], 
            text_content=row[2],
            embedding=json.loads(row[3]) if row[3] else [],
            raw_state=row[4],
            processed_state=row[5],
            strength=row[6],
            source_type=row[7] if len(row) > 7 else None
        )


@dataclass 
class ComparisonResult:
    """Single node-to-node comparison result - mirrors your log format."""
    sample_node_id: int
    comparison_node_id: int
    
    # Semantic metrics (matches your collect_sig.py format)
    embedding_similarity: float
    text_relevance: float
    semantic_density_sample: float
    semantic_density_comparison: float
    
    # Component breakdown for sample node
    sample_components: Dict[str, float]
    comparison_components: Dict[str, float]
    
    # Meta information
    database_name: str
    embedding_provider: str
    calculator_config: str
    timestamp: float


@dataclass
class TestConfiguration:
    """Test configuration for systematic experiments."""
    
    # Sampling parameters
    sample_size: int = 50           # Number of sample nodes per database
    comparisons_per_sample: int = 100  # Number of comparisons per sample node
    
    # Database paths
    database_paths: List[str] = None
    
    # Models to test
    embedding_providers: List[str] = None  # Provider aliases
    calculator_configs: List[str] = None   # Calculator configuration names
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Output configuration
    output_dir: str = "test_results"
    generate_detailed_logs: bool = True
    
    def __post_init__(self):
        if self.database_paths is None:
            self.database_paths = []
        if self.embedding_providers is None:
            self.embedding_providers = ["stella", "all_mpnet"]
        if self.calculator_configs is None:
            self.calculator_configs = ["current"]


class DatabaseLoader:
    """Loads memory nodes from your existing databases."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_name = Path(db_path).stem
    
    def load_memory_nodes(self, limit: Optional[int] = None) -> List[MemoryNode]:
        """
        Load memory nodes from cognitive_memory_nodes table.
        Uses exact schema from your db/schema.py.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query matches your schema structure
            query = """
            SELECT 
                id, timestamp, text_content, embedding, 
                raw_state, processed_state, strength,
                formation_source
            FROM cognitive_memory_nodes 
            WHERE ghosted = FALSE 
            AND text_content IS NOT NULL 
            AND text_content != ''
            AND embedding IS NOT NULL
            ORDER BY strength DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            nodes = []
            for row in rows:
                try:
                    node = MemoryNode.from_db_row(row)
                    # Validate embedding exists and is valid
                    if node.embedding and len(node.embedding) > 0:
                        nodes.append(node)
                except Exception as e:
                    print(f"Warning: Failed to parse node {row[0]}: {e}")
                    continue
            
            conn.close()
            print(f"Loaded {len(nodes)} nodes from {self.db_name}")
            return nodes
            
        except Exception as e:
            print(f"Error loading from {self.db_path}: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database for analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Total nodes
            cursor.execute("SELECT COUNT(*) FROM cognitive_memory_nodes WHERE ghosted = FALSE")
            stats['total_nodes'] = cursor.fetchone()[0]
            
            # Nodes with embeddings
            cursor.execute("""
                SELECT COUNT(*) FROM cognitive_memory_nodes 
                WHERE ghosted = FALSE AND embedding IS NOT NULL
            """)
            stats['nodes_with_embeddings'] = cursor.fetchone()[0]
            
            # Strength distribution
            cursor.execute("""
                SELECT AVG(strength), MIN(strength), MAX(strength) 
                FROM cognitive_memory_nodes WHERE ghosted = FALSE
            """)
            strength_stats = cursor.fetchone()
            stats['strength'] = {
                'mean': strength_stats[0],
                'min': strength_stats[1], 
                'max': strength_stats[2]
            }
            
            # Formation sources
            cursor.execute("""
                SELECT formation_source, COUNT(*) 
                FROM cognitive_memory_nodes 
                WHERE ghosted = FALSE 
                GROUP BY formation_source
            """)
            stats['formation_sources'] = dict(cursor.fetchall())
            
            conn.close()
            return stats
            
        except Exception as e:
            print(f"Error getting stats from {self.db_path}: {e}")
            return {}


class SemanticTestHarness:
    """
    Main test harness for systematic memory semantic analysis.
    Replicates the comparison flow and generates metrics identical to collect_*.py.
    
    Optimized with aggressive caching to avoid recomputing embeddings and reloading models.
    """
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.results: List[ComparisonResult] = []
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance optimization: Provider and embedding caches
        self._provider_cache: Dict[str, EmbeddingProvider] = {}
        self._embedding_cache: Dict[str, List[float]] = {}  # text -> embedding
        self._calculator_cache: Dict[str, ParameterizedSemanticCalculator] = {}
        
        self._semantic_density_cache: Dict[str, Dict[str, float]] = {}  # text -> density_result
        
        print(f"Initialized test harness with {config.sample_size} samples, "
              f"{config.comparisons_per_sample} comparisons each")
        print("✓ Smart on-demand caching enabled (embeddings + semantic densities)")
        print("✓ Only computes what's needed for sampled nodes")
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Run full experiment across all databases, providers, and configurations.
        Returns comprehensive analysis results.
        """
        print("=" * 80)
        print("STARTING COMPREHENSIVE SEMANTIC ANALYSIS EXPERIMENT")
        print("=" * 80)
        
        start_time = time.time()
        experiment_results = {
            'config': asdict(self.config),
            'database_stats': {},
            'comparison_results': [],
            'analysis_summary': {},
            'timing': {}
        }
        
        # Load all databases with progress
        print("Loading databases...")
        databases = {}
        for db_path in progress_bar(self.config.database_paths, desc="Loading DBs", disable=False):
            loader = DatabaseLoader(db_path)
            nodes = loader.load_memory_nodes()
            databases[loader.db_name] = {
                'nodes': nodes,
                'stats': loader.get_database_stats()
            }
            experiment_results['database_stats'][loader.db_name] = databases[loader.db_name]['stats']
        
        # OPTIMIZATION: Pre-initialize all providers and calculators
        print("Initializing embedding providers...")
        self._initialize_providers_and_calculators()
        
        total_configurations = (
            len(self.config.database_paths) * 
            len(self.config.embedding_providers) * 
            len(self.config.calculator_configs)
        )
        
        total_comparisons = (
            total_configurations *
            self.config.sample_size * 
            self.config.comparisons_per_sample
        )
        
        # Calculate actual unique texts needed (much smaller with stable comparison sets)
        max_unique_texts_needed = total_configurations * (
            self.config.sample_size + self.config.comparisons_per_sample  # No overlap since comparison set is stable
        )
        
        print(f"Will run {total_configurations} configurations with {total_comparisons:,} total comparisons")
        print(f"Smart caching: Will only process ~{max_unique_texts_needed} texts with stable comparison sets")
        
        # Create nested progress bars for configurations
        config_combinations = []
        for db_name in databases.keys():
            if not databases[db_name]['nodes']:
                print(f"Warning: No valid nodes in {db_name}, skipping")
                continue
            for provider_alias in self.config.embedding_providers:
                for calc_config in self.config.calculator_configs:
                    config_combinations.append((db_name, provider_alias, calc_config))
        
        # Run comparisons across all configurations with progress
        comparison_count = 0
        
        for db_name, provider_alias, calc_config in progress_bar(
            config_combinations, 
            desc="Configurations", 
            disable=False
        ):
            print(f"\n--- Testing {db_name} | {provider_alias} | {calc_config} ---")
            
            batch_results = self._run_comparison_batch_optimized(
                db_name=db_name,
                nodes=databases[db_name]['nodes'],
                provider_alias=provider_alias,
                calc_config=calc_config
            )
            
            self.results.extend(batch_results)
            comparison_count += len(batch_results)
            
            print(f"Completed {len(batch_results)} comparisons "
                  f"({comparison_count:,}/{total_comparisons:,})")
        
        # Generate analysis
        print("\nGenerating analysis...")
        experiment_results['comparison_results'] = [asdict(r) for r in self.results]
        experiment_results['analysis_summary'] = self._generate_analysis_summary()
        experiment_results['timing'] = {
            'total_duration': time.time() - start_time,
            'comparisons_per_second': len(self.results) / (time.time() - start_time) if len(self.results) > 0 else 0
        }
        
        # Save results
        print("Saving results...")
        self._save_experiment_results(experiment_results)
        
        print(f"\n" + "=" * 80)
        print(f"EXPERIMENT COMPLETE")
        print(f"Total comparisons: {len(self.results):,}")
        print(f"Duration: {experiment_results['timing']['total_duration']:.1f}s")
        print(f"Rate: {experiment_results['timing']['comparisons_per_second']:.1f} comparisons/sec")
        print(f"Cache stats: {len(self._embedding_cache)} embeddings cached")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80)
        
        return experiment_results
    
    def _initialize_providers_and_calculators(self) -> None:
        """Pre-initialize all providers and calculators to avoid repeated loading."""
        
        for provider_alias in progress_bar(
            self.config.embedding_providers, 
            desc="Loading models", 
            disable=False
        ):
            print(f"  Loading {provider_alias}...")
            try:
                provider = self._create_embedding_provider(provider_alias)
                self._provider_cache[provider_alias] = provider
                
                # Pre-initialize calculators for this provider
                for calc_config in self.config.calculator_configs:
                    calc_key = f"{provider_alias}_{calc_config}"
                    calculator = self._create_calculator(calc_config, provider)
                    self._calculator_cache[calc_key] = calculator
                    
                print(f"  ✓ {provider_alias} loaded ({provider.model_name})")
                
            except Exception as e:
                print(f"  ✗ Failed to load {provider_alias}: {e}")
                raise
    
    def _get_cached_semantic_density(self, text: str) -> Dict[str, float]:
        """Get cached semantic density or compute if not cached."""
        if text in self._semantic_density_cache:
            return self._semantic_density_cache[text]
        
        # Fallback: compute on demand (shouldn't happen if pre-compute worked)
        calc_key = list(self._calculator_cache.keys())[0]
        calculator = self._calculator_cache[calc_key]
        
        try:
            density_result = calculator.calculate_semantic_density(text)
            self._semantic_density_cache[text] = density_result
            return density_result
        except Exception as e:
            print(f"Warning: Failed to compute semantic density on demand: {e}")
            return {
                'components': {
                    'semantic_cohesion': 0.0,
                    'ne_density': 0.0,
                    'abstraction_level': 0.0,
                    'logical_complexity': 0.0,
                    'conceptual_bridging': 0.0,
                    'information_density': 0.0
                },
                'raw_density': 0.0,
                'transformed_density': 0.0,
                'weights': {}
            }
    
    def _get_cached_embedding(self, text: str, provider_alias: str) -> List[float]:
        """Get cached embedding or compute if not cached."""
        cache_key = f"{provider_alias}:{text}"
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Fallback: compute on demand (shouldn't happen if pre-compute worked)
        provider = self._provider_cache[provider_alias]
        try:
            embedding = provider.encode(text)
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Warning: Failed to encode text on demand: {e}")
            return [0.0] * provider.embedding_dimension
    
    def _run_comparison_batch_optimized(
        self, 
        db_name: str,
        nodes: List[MemoryNode],
        provider_alias: str, 
        calc_config: str
    ) -> List[ComparisonResult]:
        """Optimized comparison batch using cached providers and stable comparison sets."""
        
        # Get cached provider and calculator (no reloading!)
        calc_key = f"{provider_alias}_{calc_config}"
        calculator = self._calculator_cache[calc_key]
        provider = self._provider_cache[provider_alias]
        
        # Sample nodes for testing
        sample_nodes = self._sample_nodes(nodes, self.config.sample_size)
        
        # OPTIMIZATION: Sample comparison nodes ONCE for the entire batch (not per sample)
        # This massively improves cache efficiency and reduces redundant processing
        comparison_candidates = [n for n in nodes if n.id not in {s.id for s in sample_nodes}]
        comparison_nodes = self._sample_nodes(comparison_candidates, self.config.comparisons_per_sample)
        
        batch_results = []
        
        print(f"  Processing {len(sample_nodes)} samples against {len(comparison_nodes)} comparison nodes (using cached {provider_alias})...")
        
        # Progress bar for sample nodes
        for i, sample_node in enumerate(progress_bar(
            sample_nodes, 
            desc=f"  Samples ({provider_alias})", 
            disable=False
        )):
            # Run comparisons against the SAME set of comparison nodes
            for comp_node in comparison_nodes:
                try:
                    result = self._compare_nodes_optimized(
                        sample_node=sample_node,
                        comparison_node=comp_node,
                        calculator=calculator,
                        provider=provider,
                        provider_alias=provider_alias,
                        db_name=db_name,
                        calc_config=calc_config
                    )
                    batch_results.append(result)
                    
                except Exception as e:
                    print(f"Warning: Comparison failed between {sample_node.id} and {comp_node.id}: {e}")
                    continue
        
        return batch_results
    
    def _compare_nodes_optimized(
        self,
        sample_node: MemoryNode,
        comparison_node: MemoryNode,
        calculator: ParameterizedSemanticCalculator,
        provider: EmbeddingProvider,
        provider_alias: str,
        db_name: str,
        calc_config: str
    ) -> ComparisonResult:
        """
        Optimized node comparison using on-demand caching.
        Only computes what we need, when we need it.
        """
        
        # Get embeddings on-demand (cached if already computed)
        sample_embedding = self._get_cached_embedding(sample_node.text_content, provider_alias)
        comp_embedding = self._get_cached_embedding(comparison_node.text_content, provider_alias)
        
        # Get semantic densities on-demand (cached if already computed)
        sample_density_result = self._get_cached_semantic_density(sample_node.text_content)
        comp_density_result = self._get_cached_semantic_density(comparison_node.text_content)
        
        # Calculate similarity metrics using cached embeddings
        similarity_metrics = calculator.calculate_similarity_metrics(
            text_content=sample_node.text_content,
            embedding=sample_embedding,
            query_text=comparison_node.text_content,
            query_embedding=comp_embedding
        )
        
        return ComparisonResult(
            sample_node_id=sample_node.id,
            comparison_node_id=comparison_node.id,
            
            # Core similarity metrics
            embedding_similarity=similarity_metrics.get('embedding_similarity', 0.0),
            text_relevance=similarity_metrics.get('text_relevance', 0.0), 
            semantic_density_sample=sample_density_result['transformed_density'],
            semantic_density_comparison=comp_density_result['transformed_density'],
            
            # Component breakdowns
            sample_components=sample_density_result['components'],
            comparison_components=comp_density_result['components'],
            
            # Meta information
            database_name=db_name,
            embedding_provider=provider.model_name,
            calculator_config=calc_config,
            timestamp=time.time()
        )
    
    def _sample_nodes(self, nodes: List[MemoryNode], sample_size: int) -> List[MemoryNode]:
        """Sample nodes randomly but deterministically."""
        if len(nodes) <= sample_size:
            return nodes.copy()
        return random.sample(nodes, sample_size)
    
    def _create_embedding_provider(self, provider_alias: str) -> EmbeddingProvider:
        """Create embedding provider by alias (cached version)."""
        if provider_alias in self._provider_cache:
            return self._provider_cache[provider_alias]
        
        # Import here to avoid circular dependencies
        from embedding_providers import create_provider_by_alias
        provider = create_provider_by_alias(provider_alias)
        self._provider_cache[provider_alias] = provider
        return provider
    
    def _create_calculator(self, calc_config: str, embedding_provider: EmbeddingProvider) -> ParameterizedSemanticCalculator:
        """Create semantic calculator by configuration name."""
        calc_key = f"{embedding_provider.model_name}_{calc_config}"
        if calc_key in self._calculator_cache:
            return self._calculator_cache[calc_key]
        
        calculator = None
        
        # Try parameterized calculator system first
        if PARAMETERIZED_CALCULATOR_AVAILABLE:
            try:
                calculator = create_calculator_for_testing(calc_config, embedding_provider)
                print(f"  ✓ Created parameterized calculator: {calc_config}")
            except Exception as e:
                print(f"  ⚠ Parameterized calculator failed for {calc_config}: {e}")
                calculator = None
        
        # Final fallback
        if calculator is None:
            available_configs = self._get_available_calculator_configs()
            raise ValueError(
                f"Failed to create calculator '{calc_config}'. "
                f"Available: {available_configs}. "
                f"Parameterized system available: {PARAMETERIZED_CALCULATOR_AVAILABLE}. "
            )
        
        self._calculator_cache[calc_key] = calculator
        return calculator
    
    def _get_available_calculator_configs(self) -> List[str]:
        """Get list of available calculator configurations."""
        available = []
        
        if PARAMETERIZED_CALCULATOR_AVAILABLE:
            try:
                available.extend(CalculatorFactory.get_available_configs())
            except Exception:
                pass
        
        return available if available else ["none_available"]
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """
        Generate analysis summary matching collect_*.py output format.
        """
        summary = {
            'total_comparisons': len(self.results)  # Always include this field
        }
        
        if not self.results:
            summary.update({
                'error': 'No successful comparisons completed',
                'embedding_similarity_analysis': {'error': 'No data'},
                'semantic_density_analysis': {'error': 'No data'},
                'component_discrimination_analysis': {'error': 'No data'},
                'provider_comparison': {'error': 'No data'},
                'database_comparison': {'error': 'No data'}
            })
            return summary
        
        # Generate full analysis for successful results
        summary.update({
            'embedding_similarity_analysis': self._analyze_embedding_similarities(),
            'semantic_density_analysis': self._analyze_semantic_densities(),
            'component_discrimination_analysis': self._analyze_component_discrimination(),
            'provider_comparison': self._compare_providers(),
            'database_comparison': self._compare_databases()
        })
        
        return summary
    
    def _analyze_embedding_similarities(self) -> Dict[str, Any]:
        """Analyze embedding similarity distribution - mirrors collect_sig.py."""
        similarities = [r.embedding_similarity for r in self.results]
        
        if not similarities:
            return {"error": "No similarity data"}
        
        return {
            'mean': statistics.mean(similarities),
            'std_dev': statistics.stdev(similarities) if len(similarities) > 1 else 0.0,
            'min': min(similarities),
            'max': max(similarities),
            'range': max(similarities) - min(similarities),
            'coefficient_of_variation': statistics.stdev(similarities) / statistics.mean(similarities) if statistics.mean(similarities) != 0 else 0.0,
            'quartiles': {
                'q1': np.percentile(similarities, 25),
                'q2': np.percentile(similarities, 50),
                'q3': np.percentile(similarities, 75)
            }
        }
    
    def _analyze_semantic_densities(self) -> Dict[str, Any]:
        """Analyze semantic density transformation - mirrors collect_density.py."""
        sample_densities = [r.semantic_density_sample for r in self.results]
        
        if not sample_densities:
            return {"error": "No density data"}
        
        return {
            'distribution': {
                'mean': statistics.mean(sample_densities),
                'std_dev': statistics.stdev(sample_densities) if len(sample_densities) > 1 else 0.0,
                'range': max(sample_densities) - min(sample_densities),
                'coefficient_of_variation': statistics.stdev(sample_densities) / statistics.mean(sample_densities) if statistics.mean(sample_densities) != 0 else 0.0
            },
            'content_distribution': {
                'simple_content': len([d for d in sample_densities if d < 0.3]),
                'medium_content': len([d for d in sample_densities if 0.3 <= d < 0.7]),
                'complex_content': len([d for d in sample_densities if d >= 0.7])
            }
        }
    
    def _analyze_component_discrimination(self) -> Dict[str, Any]:
        """Analyze discrimination power of each component."""
        component_analysis = {}
        
        # Analyze each component across all results
        component_names = ['semantic_cohesion', 'ne_density', 'abstraction_level', 
                          'logical_complexity', 'conceptual_bridging', 'information_density']
        
        for comp_name in component_names:
            values = []
            for result in self.results:
                if comp_name in result.sample_components:
                    values.append(result.sample_components[comp_name])
            
            if values:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                component_analysis[comp_name] = {
                    'mean': mean_val,
                    'std_dev': std_val,
                    'range': max(values) - min(values),
                    'coefficient_of_variation': std_val / mean_val if mean_val != 0 else 0.0,
                    'sample_count': len(values)
                }
        
        return component_analysis
    
    def _compare_providers(self) -> Dict[str, Any]:
        """Compare performance across embedding providers."""
        provider_comparison = {}
        
        providers = set(r.embedding_provider for r in self.results)
        
        for provider in providers:
            provider_results = [r for r in self.results if r.embedding_provider == provider]
            
            if provider_results:
                similarities = [r.embedding_similarity for r in provider_results]
                densities = [r.semantic_density_sample for r in provider_results]
                
                provider_comparison[provider] = {
                    'comparison_count': len(provider_results),
                    'embedding_similarity': {
                        'mean': statistics.mean(similarities),
                        'std_dev': statistics.stdev(similarities) if len(similarities) > 1 else 0.0,
                        'coefficient_of_variation': statistics.stdev(similarities) / statistics.mean(similarities) if statistics.mean(similarities) != 0 else 0.0
                    },
                    'semantic_density': {
                        'mean': statistics.mean(densities),
                        'std_dev': statistics.stdev(densities) if len(densities) > 1 else 0.0,
                        'coefficient_of_variation': statistics.stdev(densities) / statistics.mean(densities) if statistics.mean(densities) != 0 else 0.0
                    }
                }
        
        return provider_comparison
    
    def _compare_databases(self) -> Dict[str, Any]:
        """Compare performance across different databases."""
        database_comparison = {}
        
        databases = set(r.database_name for r in self.results)
        
        for db_name in databases:
            db_results = [r for r in self.results if r.database_name == db_name]
            
            if db_results:
                similarities = [r.embedding_similarity for r in db_results]
                densities = [r.semantic_density_sample for r in db_results]
                
                database_comparison[db_name] = {
                    'comparison_count': len(db_results),
                    'embedding_similarity': {
                        'mean': statistics.mean(similarities),
                        'std_dev': statistics.stdev(similarities) if len(similarities) > 1 else 0.0
                    },
                    'semantic_density': {
                        'mean': statistics.mean(densities),
                        'std_dev': statistics.stdev(densities) if len(densities) > 1 else 0.0
                    }
                }
        
        return database_comparison
    
    def _save_experiment_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive experiment results."""
        
        # Save main results file
        results_file = self.output_dir / f"experiment_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed comparison data (for collect_*.py compatibility)
        if self.config.generate_detailed_logs:
            self._generate_detailed_logs()
        
        print(f"Results saved to {results_file}")
    
    def _generate_detailed_logs(self) -> None:
        """Generate detailed logs in collect_*.py compatible format."""
        
        # Generate significance analysis logs (collect_sig.py format)
        sig_log_file = self.output_dir / "significance_debug.log"
        with open(sig_log_file, 'w') as f:
            for result in self.results:
                # Mimic your log format for collect_sig.py compatibility
                f.write(f"[{result.timestamp}] [NOVELTY_DEBUG] Processing 1 metric entries\n")
                f.write(f"[{result.timestamp}] [NOVELTY_DEBUG] Entry 1: "
                       f"emb_sim={result.embedding_similarity:.3f}, "
                       f"text_rel={result.text_relevance:.3f}, "
                       f"density={result.semantic_density_sample:.3f}, "
                       f"novelty={result.semantic_density_sample:.3f}\n")
                f.write(f"[{result.timestamp}] [NOVELTY_DEBUG] Final novelty: {result.semantic_density_sample:.3f} from 1 scores\n")
                f.write(f"[{result.timestamp}] Memory Info: Significance evaluation complete for {result.database_name}: {result.semantic_density_sample:.3f}\n")
        
        # Generate density analysis logs (collect_density.py format)
        density_log_file = self.output_dir / "density_debug.log"
        with open(density_log_file, 'w') as f:
            for result in self.results:
                # Mimic your density log format
                f.write(f"[{result.timestamp}] [DENSITY_DEBUG] Semantic Cohesion: {result.sample_components['semantic_cohesion']:.3f}, "
                       f"NE Density: {result.sample_components['ne_density']:.3f}, "
                       f"Abstraction Level: {result.sample_components['abstraction_level']:.3f}, "
                       f"Logical Complexity: {result.sample_components['logical_complexity']:.3f}, "
                       f"Conceptual Bridging: {result.sample_components['conceptual_bridging']:.3f}, "
                       f"Information Density: {result.sample_components['information_density']:.3f}\n")
                f.write(f"[{result.timestamp}] [DENSITY_DEBUG] Density after transformation: {result.semantic_density_sample:.3f}\n")
        
        print(f"Generated compatible log files: {sig_log_file}, {density_log_file}")


# Convenience function for quick experiments
def run_quick_experiment(
    database_paths: List[str],
    sample_size: int = 20,
    comparisons_per_sample: int = 50,
    providers: List[str] = None
) -> Dict[str, Any]:
    """Run a quick experiment with sensible defaults and optimizations."""
    
    config = TestConfiguration(
        sample_size=sample_size,
        comparisons_per_sample=comparisons_per_sample,
        database_paths=database_paths,
        embedding_providers=providers or ["stella", "all_mpnet"],
        calculator_configs=["current"],
        output_dir=f"quick_test_{int(time.time())}"
    )
    
    harness = SemanticTestHarness(config)
    return harness.run_full_experiment()