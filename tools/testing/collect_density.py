#!/usr/bin/env python3
"""
Density Transformation Analysis Script

Analyzes the semantic density debug logs to evaluate the effectiveness
of the exponential transformation and identify remaining bottlenecks.
"""

import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

@dataclass
class DensityEntry:
    """Single density calculation entry with all components."""
    timestamp: str
    
    # Individual components
    cohesion: float
    ne_density: float
    conceptual_surprise: float
    logical_complexity: float
    conceptual_bridging: float
    information_density: float
    
    # Transformation data
    raw_density: Optional[float] = None
    enhanced_density: Optional[float] = None
    transform_method: Optional[str] = None
    
    # Context
    entry_id: Optional[str] = None
    evaluation_type: Optional[str] = None

@dataclass
class ComponentAnalysis:
    """Analysis of a single density component."""
    name: str
    values: List[float]
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    range_span: float
    coefficient_of_variation: float
    quartiles: Dict[str, float]
    
    @classmethod
    def from_values(cls, name: str, values: List[float]) -> 'ComponentAnalysis':
        if not values:
            return cls(name, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        min_val = min(values)
        max_val = max(values)
        range_span = max_val - min_val
        cv = std_val / mean_val if mean_val != 0 else 0.0
        
        quartiles = {
            'q1': sorted_vals[n // 4] if n > 3 else min_val,
            'q2': median_val,
            'q3': sorted_vals[3 * n // 4] if n > 3 else max_val
        }
        
        return cls(
            name=name,
            values=values,
            mean=mean_val,
            median=median_val,
            std_dev=std_val,
            min_val=min_val,
            max_val=max_val,
            range_span=range_span,
            coefficient_of_variation=cv,
            quartiles=quartiles
        )

class DensityAnalyzer:
    """Analyzes density transformation debug logs."""
    
    def __init__(self):
        self.entries: List[DensityEntry] = []
        self.parsing_context = {}
        
        # Regex patterns for parsing
        self.patterns = {
            # Component breakdown - match actual log format
            'components': re.compile(
                r'\[DENSITY_DEBUG\] Semantic Cohesion: ([0-9.]+),\s*'
                r'NE Density: ([0-9.]+),\s*'
                r'Conceptual Surprise: ([0-9.]+),\s*'
                r'Logical Complexity: ([0-9.]+),\s*'
                r'Conceptual Bridging: ([0-9.]+),\s*'
                r'Information Density: ([0-9.]+)'
            ),
            
            # Raw density before transformation
            'raw_density': re.compile(
                r'\[DENSITY_DEBUG\] Raw density before transformation: ([0-9.]+)'
            ),
            
            # Enhanced density after transformation
            'enhanced_density': re.compile(
                r'\[DENSITY_DEBUG\] Enhanced density after transformation: ([0-9.]+)'
            ),
            
            # Transformation details
            'transform_detail': re.compile(
                r'\[DENSITY_TRANSFORM\] Raw: ([0-9.]+) → Enhanced: ([0-9.]+)'
            ),
            
            # Distribution logging
            'distribution': re.compile(
                r'\[NOVELTY_DEBUG\] Density distribution: mean=([0-9.]+), std=([0-9.]+), range=([0-9.]+)'
            ),
            
            # Timestamp extraction
            'timestamp': re.compile(r'^\[([^\]]+)\]')
        }
    
    def parse_log_file(self, filepath: str) -> None:
        """Parse log file and extract density analysis data."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        current_entry = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Extract timestamp
            timestamp_match = self.patterns['timestamp'].search(line)
            timestamp = timestamp_match.group(1) if timestamp_match else f"line_{i}"
            
            # Check for component breakdown (starts new entry)
            components_match = self.patterns['components'].search(line)
            if components_match:
                # Save previous entry if exists
                if current_entry:
                    self.entries.append(current_entry)
                
                # Start new entry
                current_entry = DensityEntry(
                    timestamp=timestamp,
                    cohesion=float(components_match.group(1)),
                    ne_density=float(components_match.group(2)),
                    conceptual_surprise=float(components_match.group(3)),
                    logical_complexity=float(components_match.group(4)),
                    conceptual_bridging=float(components_match.group(5)),
                    information_density=float(components_match.group(6))
                )
                continue
            
            if current_entry is None:
                continue
                
            # Parse raw density
            raw_density_match = self.patterns['raw_density'].search(line)
            if raw_density_match:
                current_entry.raw_density = float(raw_density_match.group(1))
                continue
            
            # Parse enhanced density
            enhanced_density_match = self.patterns['enhanced_density'].search(line)
            if enhanced_density_match:
                current_entry.enhanced_density = float(enhanced_density_match.group(1))
                continue
            
            # Parse transformation details
            transform_match = self.patterns['transform_detail'].search(line)
            if transform_match:
                # Verify consistency
                raw_from_transform = float(transform_match.group(1))
                enhanced_from_transform = float(transform_match.group(2))
                
                if current_entry.raw_density is None:
                    current_entry.raw_density = raw_from_transform
                if current_entry.enhanced_density is None:
                    current_entry.enhanced_density = enhanced_from_transform
                    
                continue
        
        # Save final entry
        if current_entry:
            self.entries.append(current_entry)
    
    def analyze_density_transformation(self) -> Dict[str, Any]:
        """Comprehensive analysis of density transformation effectiveness."""
        if not self.entries:
            return {"error": "No density entries found"}
        
        analysis = {
            'summary': self._analyze_summary(),
            'component_analysis': self._analyze_components(),
            'transformation_effectiveness': self._analyze_transformation(),
            'discrimination_improvement': self._analyze_discrimination(),
            'bottleneck_identification': self._identify_remaining_bottlenecks(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _analyze_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_entries = len(self.entries)
        
        entries_with_raw = len([e for e in self.entries if e.raw_density is not None])
        entries_with_enhanced = len([e for e in self.entries if e.enhanced_density is not None])
        entries_with_both = len([e for e in self.entries if e.raw_density is not None and e.enhanced_density is not None])
        
        return {
            'total_entries': total_entries,
            'entries_with_raw_density': entries_with_raw,
            'entries_with_enhanced_density': entries_with_enhanced,
            'entries_with_transformation': entries_with_both,
            'data_completeness': entries_with_both / total_entries if total_entries > 0 else 0.0
        }
    
    def _analyze_components(self) -> Dict[str, ComponentAnalysis]:
        """Analyze individual density components."""
        component_data = {
            'semantic_cohesion': [e.cohesion for e in self.entries],
            'ne_density': [e.ne_density for e in self.entries],
            'conceptual_surprise': [e.conceptual_surprise for e in self.entries],
            'logical_complexity': [e.logical_complexity for e in self.entries],
            'conceptual_bridging': [e.conceptual_bridging for e in self.entries],
            'information_density': [e.information_density for e in self.entries]
        }
        
        return {
            name: ComponentAnalysis.from_values(name, values)
            for name, values in component_data.items()
        }
    
    def _analyze_transformation(self) -> Dict[str, Any]:
        """Analyze transformation effectiveness."""
        entries_with_both = [e for e in self.entries if e.raw_density is not None and e.enhanced_density is not None]
        
        if not entries_with_both:
            return {"error": "No entries with both raw and enhanced density"}
        
        raw_densities = [e.raw_density for e in entries_with_both]
        enhanced_densities = [e.enhanced_density for e in entries_with_both]
        
        # Calculate transformation metrics
        raw_analysis = ComponentAnalysis.from_values("raw_density", raw_densities)
        enhanced_analysis = ComponentAnalysis.from_values("enhanced_density", enhanced_densities)
        
        # Calculate transformation gains
        discrimination_gain = enhanced_analysis.coefficient_of_variation - raw_analysis.coefficient_of_variation
        range_expansion = enhanced_analysis.range_span - raw_analysis.range_span
        
        # Analyze transformation patterns
        transformation_pairs = list(zip(raw_densities, enhanced_densities))
        
        # Categorize transformations
        low_density_entries = [(r, e) for r, e in transformation_pairs if r < 0.4]
        medium_density_entries = [(r, e) for r, e in transformation_pairs if 0.4 <= r < 0.6]
        high_density_entries = [(r, e) for r, e in transformation_pairs if r >= 0.6]
        
        return {
            'raw_density_analysis': raw_analysis,
            'enhanced_density_analysis': enhanced_analysis,
            'discrimination_gain': discrimination_gain,
            'range_expansion': range_expansion,
            'transformation_categories': {
                'low_density': {
                    'count': len(low_density_entries),
                    'avg_raw': statistics.mean([r for r, e in low_density_entries]) if low_density_entries else 0,
                    'avg_enhanced': statistics.mean([e for r, e in low_density_entries]) if low_density_entries else 0,
                    'avg_gain': statistics.mean([e - r for r, e in low_density_entries]) if low_density_entries else 0
                },
                'medium_density': {
                    'count': len(medium_density_entries),
                    'avg_raw': statistics.mean([r for r, e in medium_density_entries]) if medium_density_entries else 0,
                    'avg_enhanced': statistics.mean([e for r, e in medium_density_entries]) if medium_density_entries else 0,
                    'avg_gain': statistics.mean([e - r for r, e in medium_density_entries]) if medium_density_entries else 0
                },
                'high_density': {
                    'count': len(high_density_entries),
                    'avg_raw': statistics.mean([r for r, e in high_density_entries]) if high_density_entries else 0,
                    'avg_enhanced': statistics.mean([e for r, e in high_density_entries]) if high_density_entries else 0,
                    'avg_gain': statistics.mean([e - r for r, e in high_density_entries]) if high_density_entries else 0
                }
            }
        }
    
    def _analyze_discrimination(self) -> Dict[str, Any]:
        """Analyze discrimination improvements."""
        entries_with_both = [e for e in self.entries if e.raw_density is not None and e.enhanced_density is not None]
        
        if not entries_with_both:
            return {"error": "No entries with both densities"}
        
        raw_densities = [e.raw_density for e in entries_with_both]
        enhanced_densities = [e.enhanced_density for e in entries_with_both]
        
        # Calculate discrimination metrics
        raw_cv = statistics.stdev(raw_densities) / statistics.mean(raw_densities) if raw_densities else 0
        enhanced_cv = statistics.stdev(enhanced_densities) / statistics.mean(enhanced_densities) if enhanced_densities else 0
        
        # Calculate distribution improvements
        raw_range = max(raw_densities) - min(raw_densities)
        enhanced_range = max(enhanced_densities) - min(enhanced_densities)
        
        # Calculate effective range (90th percentile)
        raw_sorted = sorted(raw_densities)
        enhanced_sorted = sorted(enhanced_densities)
        n = len(raw_sorted)
        
        raw_effective_range = raw_sorted[int(0.9 * n)] - raw_sorted[int(0.1 * n)]
        enhanced_effective_range = enhanced_sorted[int(0.9 * n)] - enhanced_sorted[int(0.1 * n)]
        
        # Analyze separation quality
        separation_quality = self._calculate_separation_quality(enhanced_densities)
        
        return {
            'coefficient_of_variation': {
                'raw': raw_cv,
                'enhanced': enhanced_cv,
                'improvement': enhanced_cv - raw_cv,
                'improvement_percentage': ((enhanced_cv - raw_cv) / raw_cv * 100) if raw_cv > 0 else 0
            },
            'range_analysis': {
                'raw_range': raw_range,
                'enhanced_range': enhanced_range,
                'range_expansion': enhanced_range - raw_range,
                'raw_effective_range': raw_effective_range,
                'enhanced_effective_range': enhanced_effective_range,
                'effective_range_expansion': enhanced_effective_range - raw_effective_range
            },
            'separation_quality': separation_quality
        }
    
    def _calculate_separation_quality(self, values: List[float]) -> Dict[str, Any]:
        """Calculate how well the transformation separates different content types."""
        if len(values) < 3:
            return {"error": "Not enough values for separation analysis"}
        
        # Define expected ranges for content types
        simple_content = [v for v in values if v < 0.3]
        medium_content = [v for v in values if 0.3 <= v < 0.7]
        complex_content = [v for v in values if v >= 0.7]
        
        # Calculate separation metrics
        separation_score = 0.0
        
        # Good separation means distinct clusters
        if simple_content and medium_content:
            simple_max = max(simple_content)
            medium_min = min(medium_content)
            if simple_max < medium_min:
                separation_score += 0.3  # Good simple/medium separation
        
        if medium_content and complex_content:
            medium_max = max(medium_content)
            complex_min = min(complex_content)
            if medium_max < complex_min:
                separation_score += 0.3  # Good medium/complex separation
        
        # Check for appropriate distribution
        total_count = len(values)
        simple_ratio = len(simple_content) / total_count
        medium_ratio = len(medium_content) / total_count
        complex_ratio = len(complex_content) / total_count
        
        # Ideal distribution might be 30% simple, 50% medium, 20% complex
        distribution_score = 1.0 - (
            abs(simple_ratio - 0.3) + 
            abs(medium_ratio - 0.5) + 
            abs(complex_ratio - 0.2)
        ) / 3.0
        
        separation_score += distribution_score * 0.4
        
        return {
            'separation_score': separation_score,
            'content_distribution': {
                'simple_content': {'count': len(simple_content), 'ratio': simple_ratio},
                'medium_content': {'count': len(medium_content), 'ratio': medium_ratio},
                'complex_content': {'count': len(complex_content), 'ratio': complex_ratio}
            },
            'cluster_separation': {
                'simple_medium_separated': simple_content and medium_content and max(simple_content) < min(medium_content),
                'medium_complex_separated': medium_content and complex_content and max(medium_content) < min(complex_content)
            }
        }
    
    def _identify_remaining_bottlenecks(self) -> Dict[str, Any]:
        """Identify remaining bottlenecks after transformation."""
        bottlenecks = {}
        
        # Check component bottlenecks
        component_analysis = self._analyze_components()
        
        for name, analysis in component_analysis.items():
            bottleneck_score = 0.0
            issues = []
            
            # Adjust thresholds based on component type
            if name == 'ne_density':
                # NE density can be legitimately low for some content
                cv_threshold = 0.3
                range_threshold = 0.15
            elif name in ['logical_complexity', 'conceptual_bridging', 'information_density']:
                # These can be legitimately low for simple content
                cv_threshold = 0.25
                range_threshold = 0.2
            else:
                # Semantic cohesion and abstraction should have good discrimination
                cv_threshold = 0.2
                range_threshold = 0.3
            
            if analysis.coefficient_of_variation < cv_threshold:
                bottleneck_score += 0.4
                issues.append("low_variance")
            
            if analysis.range_span < range_threshold:
                bottleneck_score += 0.3
                issues.append("narrow_range")
            
            if analysis.std_dev < 0.1:
                bottleneck_score += 0.3
                issues.append("minimal_spread")
            
            if bottleneck_score > 0.4:
                bottlenecks[name] = {
                    'bottleneck_score': bottleneck_score,
                    'issues': issues,
                    'analysis': analysis
                }
        
        # Check transformation bottlenecks
        transform_analysis = self._analyze_transformation()
        if 'enhanced_density_analysis' in transform_analysis:
            enhanced_analysis = transform_analysis['enhanced_density_analysis']
            
            if enhanced_analysis.coefficient_of_variation < 0.25:
                bottlenecks['enhanced_density_overall'] = {
                    'bottleneck_score': 0.8,
                    'issues': ['insufficient_discrimination_after_transformation'],
                    'analysis': enhanced_analysis
                }
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations based on analysis."""
        recommendations = []
        
        # Analyze transformation effectiveness
        transform_analysis = self._analyze_transformation()
        discrimination_analysis = self._analyze_discrimination()
        
        # Check if transformation is working
        if 'discrimination_gain' in transform_analysis:
            discrimination_gain = transform_analysis['discrimination_gain']
            
            if discrimination_gain < 0.1:
                recommendations.append(
                    "🔧 TRANSFORMATION: Discrimination gain is low ({:.3f}). Consider more aggressive "
                    "exponential curve or different transformation method.".format(discrimination_gain)
                )
        
        # Check range expansion
        if 'range_expansion' in transform_analysis:
            range_expansion = transform_analysis['range_expansion']
            
            if range_expansion < 0.3:
                recommendations.append(
                    "📏 RANGE: Range expansion is insufficient ({:.3f}). Need more aggressive "
                    "transformation to spread values across 0.1-0.9 target range.".format(range_expansion)
                )
        
        # Check component bottlenecks
        component_analysis = self._analyze_components()
        
        for name, analysis in component_analysis.items():
            if analysis.coefficient_of_variation < 0.15:
                if name == 'semantic_cohesion':
                    recommendations.append(
                        f"⚙️ SEMANTIC COHESION: Low discrimination (CV: {analysis.coefficient_of_variation:.3f}). "
                        f"Consider adjusting sentence similarity calculation or embedding model."
                    )
                elif name == 'ne_density':
                    recommendations.append(
                        f"⚙️ NE DENSITY: Low discrimination (CV: {analysis.coefficient_of_variation:.3f}). "
                        f"Consider adjusting amplification factor or entity type filtering."
                    )
                elif name == 'conceptual_surprise':
                    recommendations.append(
                        f"⚙️ CONCEPTUAL SURPRISE: Low discrimination (CV: {analysis.coefficient_of_variation:.3f}). "
                        f"Consider adjusting concrete/abstract term detection or scoring weights."
                    )
                elif name == 'logical_complexity':
                    recommendations.append(
                        f"⚙️ LOGICAL COMPLEXITY: Low discrimination (CV: {analysis.coefficient_of_variation:.3f}). "
                        f"Consider expanding discourse marker detection or social reasoning patterns."
                    )
                elif name == 'conceptual_bridging':
                    recommendations.append(
                        f"⚙️ CONCEPTUAL BRIDGING: Low discrimination (CV: {analysis.coefficient_of_variation:.3f}). "
                        f"Consider enhancing pattern recognition or relationship bridging detection."
                    )
                elif name == 'information_density':
                    recommendations.append(
                        f"⚙️ INFORMATION DENSITY: Low discrimination (CV: {analysis.coefficient_of_variation:.3f}). "
                        f"Consider balancing technical vs social information scoring."
                    )
                else:
                    recommendations.append(
                        f"⚙️ COMPONENT: {name} has low discrimination (CV: {analysis.coefficient_of_variation:.3f}). "
                        f"Consider enhancing this component's calculation or weighting."
                    )
        
        # Check separation quality
        if 'separation_quality' in discrimination_analysis:
            separation = discrimination_analysis['separation_quality']
            
            if separation.get('separation_score', 0) < 0.5:
                recommendations.append(
                    "🎯 SEPARATION: Poor content type separation. Consider multi-tier transformation "
                    "or content-specific processing."
                )
        
        return recommendations
    
    def print_comprehensive_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print comprehensive analysis report."""
        print("=" * 80)
        print("SEMANTIC DENSITY TRANSFORMATION ANALYSIS")
        print("=" * 80)
        
        # Summary
        summary = analysis['summary']
        print(f"\nSUMMARY")
        print("-" * 40)
        print(f"Total Entries: {summary['total_entries']}")
        print(f"Entries with Raw Density: {summary['entries_with_raw_density']}")
        print(f"Entries with Enhanced Density: {summary['entries_with_enhanced_density']}")
        print(f"Complete Transformations: {summary['entries_with_transformation']}")
        print(f"Data Completeness: {summary['data_completeness']:.1%}")
        
        # Component Analysis
        print(f"\nCOMPONENT ANALYSIS")
        print("-" * 40)
        
        component_analysis = analysis['component_analysis']
        for name, comp_analysis in component_analysis.items():
            print(f"\n{name.upper()}:")
            print(f"  Mean: {comp_analysis.mean:.3f}, Std: {comp_analysis.std_dev:.3f}")
            print(f"  Range: {comp_analysis.min_val:.3f} - {comp_analysis.max_val:.3f} (span: {comp_analysis.range_span:.3f})")
            print(f"  CV: {comp_analysis.coefficient_of_variation:.3f}")
            
            # Highlight issues
            issues = []
            if comp_analysis.coefficient_of_variation < 0.2:
                issues.append("LOW_VARIANCE")
            if comp_analysis.range_span < 0.3:
                issues.append("NARROW_RANGE")
            if comp_analysis.std_dev < 0.1:
                issues.append("MINIMAL_SPREAD")
            
            if issues:
                print(f"  ⚠️  ISSUES: {', '.join(issues)}")
        
        # Transformation Effectiveness
        if 'transformation_effectiveness' in analysis:
            print(f"\nTRANSFORMATION EFFECTIVENESS")
            print("-" * 40)
            
            transform = analysis['transformation_effectiveness']
            
            if 'raw_density_analysis' in transform:
                raw = transform['raw_density_analysis']
                enhanced = transform['enhanced_density_analysis']
                
                print(f"RAW DENSITY:")
                print(f"  Mean: {raw.mean:.3f}, Std: {raw.std_dev:.3f}, CV: {raw.coefficient_of_variation:.3f}")
                print(f"  Range: {raw.min_val:.3f} - {raw.max_val:.3f} (span: {raw.range_span:.3f})")
                
                print(f"\nENHANCED DENSITY:")
                print(f"  Mean: {enhanced.mean:.3f}, Std: {enhanced.std_dev:.3f}, CV: {enhanced.coefficient_of_variation:.3f}")
                print(f"  Range: {enhanced.min_val:.3f} - {enhanced.max_val:.3f} (span: {enhanced.range_span:.3f})")
                
                print(f"\nIMPROVEMENT:")
                print(f"  Discrimination Gain: {transform.get('discrimination_gain', 0):.3f}")
                print(f"  Range Expansion: {transform.get('range_expansion', 0):.3f}")
                
                # Show transformation categories
                if 'transformation_categories' in transform:
                    cats = transform['transformation_categories']
                    print(f"\nTRANSFORMATION BY CATEGORY:")
                    for category, data in cats.items():
                        if data['count'] > 0:
                            print(f"  {category.upper()}: {data['count']} entries")
                            print(f"    Raw: {data['avg_raw']:.3f} → Enhanced: {data['avg_enhanced']:.3f} "
                                  f"(gain: {data['avg_gain']:.3f})")
        
        # Discrimination Improvement
        if 'discrimination_improvement' in analysis:
            print(f"\nDISCRIMINATION IMPROVEMENT")
            print("-" * 40)
            
            discrimination = analysis['discrimination_improvement']
            
            if 'coefficient_of_variation' in discrimination:
                cv = discrimination['coefficient_of_variation']
                print(f"Coefficient of Variation:")
                print(f"  Raw: {cv['raw']:.3f} → Enhanced: {cv['enhanced']:.3f}")
                print(f"  Improvement: {cv['improvement']:.3f} ({cv['improvement_percentage']:.1f}%)")
            
            if 'range_analysis' in discrimination:
                ranges = discrimination['range_analysis']
                print(f"\nRange Analysis:")
                print(f"  Total Range: {ranges['raw_range']:.3f} → {ranges['enhanced_range']:.3f}")
                print(f"  Effective Range: {ranges['raw_effective_range']:.3f} → {ranges['enhanced_effective_range']:.3f}")
                print(f"  Range Expansion: {ranges['range_expansion']:.3f}")
            
            if 'separation_quality' in discrimination:
                separation = discrimination['separation_quality']
                print(f"\nSeparation Quality:")
                print(f"  Separation Score: {separation.get('separation_score', 0):.3f}")
                
                if 'content_distribution' in separation:
                    dist = separation['content_distribution']
                    print(f"  Content Distribution:")
                    print(f"    Simple: {dist['simple_content']['count']} ({dist['simple_content']['ratio']:.1%})")
                    print(f"    Medium: {dist['medium_content']['count']} ({dist['medium_content']['ratio']:.1%})")
                    print(f"    Complex: {dist['complex_content']['count']} ({dist['complex_content']['ratio']:.1%})")
        
        # Bottlenecks
        if 'bottleneck_identification' in analysis:
            print(f"\nREMAINING BOTTLENECKS")
            print("-" * 40)
            
            bottlenecks = analysis['bottleneck_identification']
            
            if bottlenecks:
                for name, bottleneck in bottlenecks.items():
                    score = bottleneck['bottleneck_score']
                    issues = bottleneck['issues']
                    
                    severity = "🔴 CRITICAL" if score >= 0.7 else "🟡 MODERATE" if score >= 0.4 else "🟢 MINOR"
                    
                    print(f"\n{name}: {severity} (score: {score:.2f})")
                    print(f"  Issues: {', '.join(issues)}")
                    
                    if hasattr(bottleneck['analysis'], 'coefficient_of_variation'):
                        analysis_obj = bottleneck['analysis']
                        print(f"  CV: {analysis_obj.coefficient_of_variation:.3f}, "
                              f"Range: {analysis_obj.range_span:.3f}, "
                              f"Std: {analysis_obj.std_dev:.3f}")
            else:
                print("✅ No critical bottlenecks identified!")
        
        # Recommendations
        if 'recommendations' in analysis:
            print(f"\nRECOMMENDATIONS")
            print("-" * 40)
            
            recommendations = analysis['recommendations']
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. {rec}")
            else:
                print("✅ No specific recommendations - transformation appears effective!")

def main():
    if len(sys.argv) < 2:
        print("Usage: python density_analyzer.py <log_file_path>")
        print("\nAnalyzes semantic density transformation debug logs to evaluate")
        print("the effectiveness of the exponential transformation.")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    analyzer = DensityAnalyzer()
    
    try:
        print("Parsing density transformation logs...")
        analyzer.parse_log_file(log_file)
        
        print("Analyzing transformation effectiveness...")
        analysis = analyzer.analyze_density_transformation()
        
        analyzer.print_comprehensive_analysis(analysis)
        
    except FileNotFoundError:
        print(f"Error: Could not find log file '{log_file}'")
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()