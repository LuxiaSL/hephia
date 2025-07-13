"""
testing/calculator_integration.py

Enhanced integration layer for parameterized calculators with test harness.
Updated to support performance optimization and parameter classification.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict

from semantic_calculator import (
    ParameterizedSemanticCalculator, 
    CalculatorConfiguration,
    ComponentWeights,
    TransformationParams,
    DiscriminatorConfig,
    create_baseline_config,
    create_optimized_config,
    create_comprehensive_config
)
from semantic_calculator import ParameterizedSemanticCalculator
from embedding_providers import EmbeddingProvider


class CalculatorFactory:
    """
    Enhanced factory for creating and managing different calculator configurations.
    Now includes parameter classification and performance optimization support.
    """
    
    # Predefined configurations for testing and optimization
    PRESET_CONFIGS = {
        "baseline": create_baseline_config,
        "optimized": create_optimized_config, 
        "comprehensive": create_comprehensive_config,
        
        # Experimental weight distributions
        "density_focused": lambda: CalculatorConfiguration(
            component_weights=ComponentWeights(
                ne_density=0.50,
                conceptual_surprise=0.25,
                logical_complexity=0.10,
                conceptual_bridging=0.10,
                information_density=0.05
            )
        ),
        
        "cohesion_focused": lambda: CalculatorConfiguration(
            component_weights=ComponentWeights(
                ne_density=0.30,
                conceptual_surprise=0.25,
                logical_complexity=0.15,
                conceptual_bridging=0.25,
                information_density=0.05
            )
        ),
        
        "balanced": lambda: CalculatorConfiguration(
            component_weights=ComponentWeights(
                ne_density=0.24,
                conceptual_surprise=0.24,
                logical_complexity=0.24,
                conceptual_bridging=0.19,
                information_density=0.09
            )
        ),
        
        "complexity_focused": lambda: CalculatorConfiguration(
            component_weights=ComponentWeights(
                ne_density=0.18,
                conceptual_surprise=0.18,
                logical_complexity=0.38,
                conceptual_bridging=0.18,
                information_density=0.08
            )
        ),
        
        "information_focused": lambda: CalculatorConfiguration(
            component_weights=ComponentWeights(
                ne_density=0.25,
                conceptual_surprise=0.20,
                logical_complexity=0.15,
                conceptual_bridging=0.15,
                information_density=0.25
            ),
            discriminator_config=DiscriminatorConfig(
                info_normalization=2.594,
                factual_info_weight=1.678,
                technical_info_weight=0.942,
                info_density_amplification=3.909,
            )
        ),
        
        "analysis_optimized": lambda: CalculatorConfiguration(
            component_weights=ComponentWeights(
                ne_density=0.30,
                conceptual_surprise=0.25,
                logical_complexity=0.20,
                conceptual_bridging=0.15,
                information_density=0.10
            ),
            discriminator_config=DiscriminatorConfig(
                ne_amplification_factor=17.792,
                concept_surprise_normalization=3.526,
                info_normalization=2.594,
                factual_info_weight=1.678,
                semantic_reasoning_weight=0.991,
                bridge_normalization=0.488,
                technical_info_weight=0.942,
            ),
            transformation_params=TransformationParams(
                max_output=0.927,
                high_scale=15.390,
            )
        ),
        
        "gentle_transform": lambda: CalculatorConfiguration(
            transformation_params=TransformationParams(
                low_power=2.0, mid_slope=3.0, high_scale=8.0  # Gentler curves
            )
        ),
        
        "aggressive_transform": lambda: CalculatorConfiguration(
            transformation_params=TransformationParams(
                low_power=6.0, mid_slope=8.0, high_scale=25.0  # Steeper curves
            )
        ),
        
        "fast_heuristic": lambda: CalculatorConfiguration(
            cohesion_method="word_overlap",
            ne_detection_method="heuristic", 
            complexity_method="semantic_only",
            enable_parallel_processing=True,
            max_workers=8
        ),
        
        "slow_comprehensive": lambda: CalculatorConfiguration(
            cohesion_method="embedding_similarity",
            complexity_method="hybrid",
            enable_parallel_processing=True,
            max_workers=2,
            log_calculation_details=True
        )
    }
    
    @classmethod
    def create_calculator(
        cls, 
        config_name: str, 
        embedding_provider: Optional[EmbeddingProvider] = None,
        custom_config: Optional[CalculatorConfiguration] = None
    ) -> ParameterizedSemanticCalculator:
        """
        Create calculator instance from configuration name or custom config.
        
        Args:
            config_name: Name of preset configuration or "custom"
            embedding_provider: Optional embedding provider
            custom_config: Custom configuration (required if config_name="custom")
            
        Returns:
            Calculator instance
        """
        if config_name == "custom":
            if custom_config is None:
                raise ValueError("custom_config required when config_name='custom'")
            config = custom_config
        elif config_name == "current":
            # Legacy compatibility - map to baseline
            config = cls.PRESET_CONFIGS["baseline"]()
        elif config_name in cls.PRESET_CONFIGS:
            config = cls.PRESET_CONFIGS[config_name]()
        else:
            raise ValueError(f"Unknown configuration: {config_name}. Available: {list(cls.PRESET_CONFIGS.keys())}")
        
        return ParameterizedSemanticCalculator(config, embedding_provider)
    
    @classmethod
    def create_from_vector(cls, parameter_vector: np.ndarray, 
                          embedding_provider: Optional[EmbeddingProvider] = None,
                          base_config_name: str = "baseline") -> 'ParameterizedSemanticCalculator':
        """
        Create calculator from parameter vector with ROBUST error handling.
        Updated for 37-parameter system
        """
        try:
            # Validate parameter vector length
            expected_length = 37  # 5 + 9 + 23 = 37 parameters
            if len(parameter_vector) != expected_length:
                raise ValueError(f"Parameter vector length {len(parameter_vector)} != expected {expected_length}")
            
            # Get base configuration
            if base_config_name in cls.PRESET_CONFIGS:
                base_config = cls.PRESET_CONFIGS[base_config_name]()
            else:
                base_config = CalculatorConfiguration()
            
            # Create configuration from vector using DETERMINISTIC ordering
            config = CalculatorConfiguration.from_optimization_vector(parameter_vector, base_config)
            
            # Create calculator
            calculator = ParameterizedSemanticCalculator(config, embedding_provider)
            
            return calculator
            
        except Exception as e:
            print(f"❌ Failed to create calculator from vector: {e}")
            print(f"   Vector length: {len(parameter_vector)}, expected: {expected_length}")
            # Fall back to baseline configuration
            base_config = CalculatorConfiguration()
            return ParameterizedSemanticCalculator(base_config, embedding_provider)
    
    @classmethod
    def get_parameter_bounds(cls) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds for neural optimization with enhanced classification.
        
        Returns:
            Dictionary mapping parameter names to (min, max) bounds with type metadata
        """
        return {
            # Component weights (will be normalized, so bounds are flexible)
            'ne_density_weight': (0.05, 0.50),
            'conceptual_surprise_weight': (0.05, 0.40),
            'logical_complexity_weight': (0.05, 0.40),
            'conceptual_bridging_weight': (0.05, 0.40),
            'information_density_weight': (0.01, 0.20),
            
            # Transformation parameters
            'low_threshold': (0.20, 0.50),
            'mid_threshold': (0.35, 0.60),
            'low_power': (1.5, 8.0),
            'low_scale': (0.1, 0.4),
            'mid_slope': (3.0, 10.0),
            'high_base': (0.3, 0.7),
            'high_scale': (5.0, 30.0),
            'min_output': (0.01, 0.10),
            'max_output': (0.8, 0.99),
            
            # Discriminator parameters
            'ne_amplification_factor': (5.0, 25.0),

            'abstract_boost': (0.5, 2.0),
            'concrete_boost': (0.5, 2.0),
            'length_weight': (0.5, 2.0),

            'dependency_weight': (0.5, 2.0),
            'pos_weight': (0.5, 2.0),
            'semantic_reasoning_weight': (0.5, 2.0),
            'complexity_normalization': (5.0, 20.0),

            'syntactic_bridge_weight': (1.0, 4.0),
            'semantic_bridge_weight': (1.0, 4.0),
            'entity_bridge_weight': (0.5, 2.0),
            'bridge_normalization': (0.2, 1.0),

            'technical_info_weight': (0.5, 2.0),
            'social_info_weight': (0.5, 2.0),
            'factual_info_weight': (0.5, 2.0),
            'info_density_amplification': (2.0, 8.0),
            'info_normalization': (1.0, 5.0),

            'cohesion_fallback_similarity': (0.3, 0.7),
            'min_sentences_for_cohesion': (1, 3),
            
            # Conceptual Surprise parameters
            'syntactic_surprise_weight': (0.5, 2.0),
            'semantic_role_surprise_weight': (0.5, 2.0),
            'discourse_surprise_weight': (0.5, 2.0),
            'concept_surprise_normalization': (2.0, 5.0),
        }
    
    @classmethod
    def get_parameter_metadata(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get enhanced parameter metadata including type and dependency classification.
        
        Returns:
            Dictionary with parameter type and dependency information
        """
        bounds = cls.get_parameter_bounds()
        
        # Classify parameters by embedding dependency
        embedding_independent = {
            # Amplification factors (raw signal scaling)
            'ne_amplification_factor', 'info_density_amplification',
            'concept_surprise_normalization', 'complexity_normalization'
            
            # Discriminator weights (spacy-based features)
            'dependency_weight', 'pos_weight', 'semantic_reasoning_weight',
            'technical_info_weight', 'social_info_weight', 'factual_info_weight',
            'syntactic_bridge_weight', 'semantic_bridge_weight', 'entity_bridge_weight',
            'syntactic_surprise_weight', 'semantic_role_surprise_weight', 'discourse_surprise_weight',
            
            # Transformation parameters (curve shaping)
            'low_power', 'high_scale', 'mid_slope', 'low_threshold', 'mid_threshold',
            'low_scale', 'high_base', 'min_output', 'max_output',
            
            # Normalization factors
            'bridge_normalization', 'info_normalization',
            
            # Configuration parameters
            'abstract_boost', 'concrete_boost', 'length_weight',
            'min_sentences_for_cohesion',
        }
        
        embedding_dependent = {
            # Component weights (need rebalancing per provider)
            'ne_density_weight', 'conceptual_surprise_weight',
            'logical_complexity_weight', 'conceptual_bridging_weight', 'information_density_weight',
            
            # Embedding-specific thresholds (only if using embedding_similarity method)
            'cohesion_fallback_similarity',
        }
        
        # Determine integer parameters
        integer_params = {
            'min_sentences_for_cohesion'
        }
        
        metadata = {}
        for param_name, bounds_tuple in bounds.items():
            metadata[param_name] = {
                'bounds': bounds_tuple,
                'type': 'integer' if param_name in integer_params else 'real',
                'embedding_dependency': (
                    'independent' if param_name in embedding_independent else
                    'dependent' if param_name in embedding_dependent else
                    'unknown'
                ),
                'component_mapping': cls._get_component_mapping(param_name)
            }
        
        return metadata
    
    @classmethod
    def _get_component_mapping(cls, param_name: str) -> Optional[str]:
        """Map parameter to its primary component."""
        component_mapping = {
            # NE density
            'ne_density_weight': 'ne_density',
            'ne_amplification_factor': 'ne_density',
            
            # Conceptual surprise
            'conceptual_surprise_weight': 'conceptual_surprise',
            'syntactic_surprise_weight': 'conceptual_surprise',
            'semantic_role_surprise_weight': 'conceptual_surprise',
            'discourse_surprise_weight': 'conceptual_surprise',
            'concept_surprise_normalization': 'conceptual_surprise',
            
            # Logical complexity
            'logical_complexity_weight': 'logical_complexity',
            'dependency_weight': 'logical_complexity',
            'pos_weight': 'logical_complexity',
            'semantic_reasoning_weight': 'logical_complexity',
            'complexity_normalization': 'logical_complexity',
            
            # Conceptual bridging
            'conceptual_bridging_weight': 'conceptual_bridging',
            'syntactic_bridge_weight': 'conceptual_bridging',
            'semantic_bridge_weight': 'conceptual_bridging',
            'entity_bridge_weight': 'conceptual_bridging',
            'bridge_normalization': 'conceptual_bridging',
            
            # Information density
            'information_density_weight': 'information_density',
            'technical_info_weight': 'information_density',
            'social_info_weight': 'information_density',
            'factual_info_weight': 'information_density',
            'info_density_amplification': 'information_density',
            'info_normalization': 'information_density',
        }
        
        return component_mapping.get(param_name)
    
    @staticmethod
    def get_focused_parameter_bounds() -> Dict[str, Tuple[float, float]]:
        """
        Get focused bounds for high-impact parameters based on statistical analysis.
        """
        # Get standard bounds
        bounds = CalculatorFactory.get_parameter_bounds()
        
        # Override with focused ranges for high-impact parameters
        focused_overrides = {
            'info_normalization': (2.0, 4.0),  # μ=2.59, correlation=0.888
            'factual_info_weight': (1.3, 2.0),  # μ=1.68, correlation=0.727
            'max_output': (0.85, 0.98),  # μ=0.93, correlation=0.576
            'concept_surprise_normalization': (2.5, 4.5),  # μ=3.53, correlation=0.540
            'ne_amplification_factor': (15.0, 22.0),  # μ=17.79, correlation=0.452
            
            # Other high-impact parameters
            'complexity_normalization': (10.0, 16.0),  # Best performers outside bounds
            'bridge_normalization': (0.2, 0.8),  # μ=0.49, correlation=0.458
            'semantic_reasoning_weight': (0.5, 1.6),  # μ=0.99, correlation=0.476
        }
        
        # Apply focused overrides
        bounds.update(focused_overrides)
        return bounds
    
    @classmethod
    def get_optimization_space_size(cls) -> int:
        """Get the size of the optimization vector."""
        baseline = cls.PRESET_CONFIGS["baseline"]()
        return len(baseline.to_optimization_vector())
    
    @classmethod
    def save_config(cls, config: CalculatorConfiguration, filepath: str) -> None:
        """Save configuration to JSON file."""
        # Convert to serializable format
        config_dict = {
            'component_weights': asdict(config.component_weights),
            'transformation_params': asdict(config.transformation_params),
            'discriminator_config': asdict(config.discriminator_config),
            'enable_caching': config.enable_caching,
            'cache_size': config.cache_size,
            'enable_parallel_processing': config.enable_parallel_processing,
            'max_workers': config.max_workers,
            'strict_validation': config.strict_validation,
            'fallback_on_errors': config.fallback_on_errors,
            'log_calculation_details': config.log_calculation_details,
            'cohesion_method': config.cohesion_method,
            'ne_detection_method': config.ne_detection_method,
            'complexity_method': config.complexity_method
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> CalculatorConfiguration:
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct configuration objects
        component_weights = ComponentWeights(**config_dict['component_weights'])
        transformation_params = TransformationParams(**config_dict['transformation_params'])
        discriminator_config = DiscriminatorConfig(**config_dict['discriminator_config'])
        
        return CalculatorConfiguration(
            component_weights=component_weights,
            transformation_params=transformation_params,
            discriminator_config=discriminator_config,
            enable_caching=config_dict.get('enable_caching', True),
            cache_size=config_dict.get('cache_size', 1000),
            enable_parallel_processing=config_dict.get('enable_parallel_processing', True),
            max_workers=config_dict.get('max_workers', 4),
            strict_validation=config_dict.get('strict_validation', True),
            fallback_on_errors=config_dict.get('fallback_on_errors', True),
            log_calculation_details=config_dict.get('log_calculation_details', False),
            cohesion_method=config_dict.get('cohesion_method', 'embedding_similarity'),
            ne_detection_method=config_dict.get('ne_detection_method', 'spacy'),
            complexity_method=config_dict.get('complexity_method', 'hybrid')
        )
    
    @classmethod
    def get_available_configs(cls) -> List[str]:
        """Get list of available preset configurations."""
        return list(cls.PRESET_CONFIGS.keys()) + ["custom", "current"]


class ConfigurationManager:
    """
    Enhanced manager for handling multiple calculator configurations and experiments.
    Now includes parameter analysis and hybrid seeding support.
    """
    
    def __init__(self, experiment_dir: str = "calculator_experiments"):
        """
        Initialize configuration manager.
        
        Args:
            experiment_dir: Directory to store configurations and results
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.configurations: Dict[str, CalculatorConfiguration] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def register_configuration(self, name: str, config: CalculatorConfiguration) -> None:
        """Register a configuration for tracking."""
        self.configurations[name] = config
        
        # Save to disk
        config_file = self.experiment_dir / f"{name}_config.json"
        CalculatorFactory.save_config(config, str(config_file))
    
    def create_experiment_config(
        self,
        name: str,
        base_config: str = "baseline",
        modifications: Optional[Dict[str, Any]] = None
    ) -> CalculatorConfiguration:
        """
        Create experimental configuration with modifications.
        
        Args:
            name: Name for the new configuration
            base_config: Base configuration name
            modifications: Dictionary of modifications to apply
            
        Returns:
            New configuration with modifications applied
        """
        # Start with base configuration
        base = CalculatorFactory.PRESET_CONFIGS[base_config]()
        
        if modifications:
            # Apply modifications
            if 'component_weights' in modifications:
                weights_dict = asdict(base.component_weights)
                weights_dict.update(modifications['component_weights'])
                base.component_weights = ComponentWeights(**weights_dict).normalize()
            
            if 'transformation_params' in modifications:
                transform_dict = asdict(base.transformation_params)
                transform_dict.update(modifications['transformation_params'])
                base.transformation_params = TransformationParams(**transform_dict)
            
            if 'discriminator_config' in modifications:
                discriminator_dict = asdict(base.discriminator_config)
                discriminator_dict.update(modifications['discriminator_config'])
                base.discriminator_config = DiscriminatorConfig(**discriminator_dict)
            
            # Apply other modifications
            for key, value in modifications.items():
                if key not in ['component_weights', 'transformation_params', 'discriminator_config']:
                    if hasattr(base, key):
                        setattr(base, key, value)
        
        # Register and return
        self.register_configuration(name, base)
        return base
    
    def analyze_parameter_transferability(
        self, 
        results1: Dict[str, Any], 
        results2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze which parameters can be transferred between optimization runs.
        
        Args:
            results1: Results from first optimization run
            results2: Results from second optimization run
            
        Returns:
            Analysis of transferable parameters
        """
        metadata = CalculatorFactory.get_parameter_metadata()
        
        # Extract best configurations from both runs
        best1 = max(results1.get('results', []), key=lambda x: x.get('reward', 0))
        best2 = max(results2.get('results', []), key=lambda x: x.get('reward', 0))
        
        if not best1 or not best2:
            return {'error': 'Insufficient data for analysis'}
        
        param_names = results1.get('parameter_names', [])
        params1 = dict(zip(param_names, best1.get('parameter_vector', [])))
        params2 = dict(zip(param_names, best2.get('parameter_vector', [])))
        
        analysis = {
            'embedding_independent_params': {},
            'embedding_dependent_params': {},
            'component_analysis': {},
            'transferability_score': 0.0
        }
        
        # Analyze embedding-independent parameters
        for param_name, meta in metadata.items():
            if param_name in params1 and param_name in params2:
                if meta['embedding_dependency'] == 'independent':
                    analysis['embedding_independent_params'][param_name] = {
                        'run1_value': params1[param_name],
                        'run2_value': params2[param_name],
                        'difference': abs(params1[param_name] - params2[param_name]),
                        'component': meta['component_mapping']
                    }
                elif meta['embedding_dependency'] == 'dependent':
                    analysis['embedding_dependent_params'][param_name] = {
                        'run1_value': params1[param_name],
                        'run2_value': params2[param_name],
                        'difference': abs(params1[param_name] - params2[param_name]),
                        'component': meta['component_mapping']
                    }
        
        # Calculate transferability score
        if analysis['embedding_independent_params']:
            avg_difference = np.mean([
                data['difference'] for data in analysis['embedding_independent_params'].values()
            ])
            analysis['transferability_score'] = max(0.0, 1.0 - avg_difference)
        
        return analysis
    
    def record_experiment_result(
        self,
        config_name: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record results from an experiment."""
        self.results[config_name] = {
            'results': results,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        # Save to disk
        results_file = self.experiment_dir / f"{config_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results[config_name], f, indent=2)
    
    def get_best_configuration(
        self,
        metric: str = "discrimination_index",
        higher_is_better: bool = True
    ) -> Optional[str]:
        """
        Get the best performing configuration based on a metric.
        
        Args:
            metric: Metric to optimize for
            higher_is_better: Whether higher values are better
            
        Returns:
            Name of best configuration or None if no results
        """
        if not self.results:
            return None
        
        best_config = None
        best_value = float('-inf') if higher_is_better else float('inf')
        
        for config_name, result_data in self.results.items():
            results = result_data['results']
            
            # Extract metric value (handle nested dictionaries)
            value = self._extract_metric_value(results, metric)
            if value is None:
                continue
            
            if higher_is_better:
                if value > best_value:
                    best_value = value
                    best_config = config_name
            else:
                if value < best_value:
                    best_value = value
                    best_config = config_name
        
        return best_config
    
    def _extract_metric_value(self, results: Dict[str, Any], metric: str) -> Optional[float]:
        """Extract metric value from nested results dictionary."""
        # Handle common metric paths
        metric_paths = {
            'discrimination_index': ['analysis_summary', 'embedding_similarity_analysis', 'coefficient_of_variation'],
            'density_cv': ['analysis_summary', 'semantic_density_analysis', 'distribution', 'coefficient_of_variation'],
            'comparisons_per_sec': ['timing', 'comparisons_per_second'],
            'total_comparisons': ['analysis_summary', 'total_comparisons']
        }
        
        if metric in metric_paths:
            path = metric_paths[metric]
        else:
            path = metric.split('.')
        
        # Navigate nested dictionary
        current = results
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return float(current) if isinstance(current, (int, float)) else None
    
    def compare_configurations(
        self,
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare all configurations across specified metrics.
        
        Args:
            metrics: List of metrics to compare
            
        Returns:
            Dictionary mapping config names to metric values
        """
        if metrics is None:
            metrics = [
                'discrimination_index',
                'density_cv',
                'comparisons_per_sec',
                'total_comparisons'
            ]
        
        comparison = {}
        
        for config_name, result_data in self.results.items():
            comparison[config_name] = {}
            
            for metric in metrics:
                value = self._extract_metric_value(result_data['results'], metric)
                comparison[config_name][metric] = value
        
        return comparison


# Integration with test harness
def create_calculator_for_testing(
    calc_config: str,
    embedding_provider: EmbeddingProvider
) -> ParameterizedSemanticCalculator:
    """
    Factory function for test harness integration.
    
    Args:
        calc_config: Configuration name or "custom:path/to/config.json"
        embedding_provider: Embedding provider instance
        
    Returns:
        Calculator instance
    """
    if calc_config.startswith("custom:"):
        # Load custom configuration from file
        config_path = calc_config[7:]  # Remove "custom:" prefix
        config = CalculatorFactory.load_config(config_path)
        return CalculatorFactory.create_calculator("custom", embedding_provider, config)
    else:
        # Use preset configuration
        return CalculatorFactory.create_calculator(calc_config, embedding_provider)


# Convenience functions for common use cases
def quick_comparison_experiment(
    database_paths: List[str],
    config_names: List[str] = None,
    embedding_provider: str = "all_minilm",
    sample_size: int = 5,
    comparisons: int = 20
) -> Dict[str, Any]:
    """
    Run quick comparison experiment across multiple calculator configurations.
    
    Args:
        database_paths: Paths to memory databases
        config_names: List of configuration names to test
        embedding_provider: Embedding provider to use
        sample_size: Number of sample nodes
        comparisons: Number of comparisons per sample
        
    Returns:
        Comparison results across configurations
    """
    if config_names is None:
        config_names = ["baseline", "density_focused", "cohesion_focused", "balanced"]
    
    # Import here to avoid circular dependency
    from test_harness import TestConfiguration, SemanticTestHarness
    
    results = {}
    
    for config_name in config_names:
        print(f"\nTesting configuration: {config_name}")
        
        test_config = TestConfiguration(
            sample_size=sample_size,
            comparisons_per_sample=comparisons,
            database_paths=database_paths,
            embedding_providers=[embedding_provider],
            calculator_configs=[config_name],
            output_dir=f"quick_comparison_{config_name}_{int(time.time())}"
        )
        
        harness = SemanticTestHarness(test_config)
        result = harness.run_full_experiment()
        results[config_name] = result
    
    return results