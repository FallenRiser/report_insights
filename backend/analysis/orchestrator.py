"""
Analysis Orchestrator

Runs all analysis upfront, then insights are generated FROM these results.
This ensures insights are grounded in actual statistical analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import asyncio

import polars as pl

from core.logging_config import insights_logger as logger
from core.data_understanding import DataUnderstanding


@dataclass
class AnalysisResults:
    """
    Container for all analysis results.
    
    Insights will be generated FROM these results,
    ensuring they are backed by actual statistical analysis.
    """
    
    # Correlation analysis
    correlations: list[dict] = field(default_factory=list)
    correlation_matrix: Optional[dict] = None
    
    # Key influencers - what drives the primary measure
    key_influencers: Optional[dict] = None
    
    # Decomposition - how measure breaks down by dimensions
    decomposition: Optional[dict] = None
    
    # Outliers per measure
    outliers: dict[str, dict] = field(default_factory=dict)
    
    # Trends per measure (if time column exists)
    trends: dict[str, dict] = field(default_factory=dict)
    
    # Distribution analysis
    distributions: dict[str, dict] = field(default_factory=dict)
    
    # High-level statistics
    statistics: dict[str, dict] = field(default_factory=dict)


class AnalysisOrchestrator:
    """
    Orchestrates all analysis in the correct order.
    
    Flow:
    1. Run correlation analysis
    2. Run key influencers (uses primary measure)
    3. Run decomposition (breaks down primary measure)
    4. Run outlier detection
    5. Run trend analysis (if time column exists)
    """
    
    def __init__(self):
        self.logger = logger
    
    async def run_all(
        self,
        df: pl.DataFrame,
        understanding: DataUnderstanding,
    ) -> AnalysisResults:
        """
        Run all analysis and return unified results.
        """
        self.logger.info("=== ANALYSIS ORCHESTRATOR STARTED ===")
        results = AnalysisResults()
        
        # Import convert_numpy for serialization safety
        from api.schemas.responses import convert_numpy
        
        # Get key columns
        measures = [m.name for m in understanding.measure_columns]
        dimensions = [g.name for g in understanding.grouping_columns]
        time_cols = [t.name for t in understanding.time_columns]
        
        primary_measure = measures[0] if measures else None
        
        self.logger.info(f"Primary measure: {primary_measure}")
        self.logger.info(f"Measures: {measures[:5]}")
        self.logger.info(f"Dimensions: {dimensions[:5]}")
        
        # 1. CORRELATION ANALYSIS
        self.logger.info("Running correlation analysis...")
        try:
            from analysis.correlations import correlation_analyzer
            if len(measures) >= 2:
                corr_matrix = correlation_analyzer.compute_correlation_matrix(
                    df, measures[:10]
                )
                results.correlations = [
                    convert_numpy(p.to_dict()) 
                    for p in corr_matrix.significant_pairs[:20]
                ]
                results.correlation_matrix = {
                    "columns": corr_matrix.columns,
                    "matrix": convert_numpy(corr_matrix.pearson_matrix),
                }
                self.logger.success(f"Found {len(results.correlations)} significant correlations")
        except Exception as e:
            self.logger.warning(f"Correlation analysis failed: {e}")
        
        # 2. KEY INFLUENCERS
        if primary_measure:
            self.logger.info(f"Running key influencers for {primary_measure}...")
            try:
                from analysis.key_influencers import key_influencers_analyzer
                ki_result = key_influencers_analyzer.analyze(
                    df, target_column=primary_measure, top_n=10
                )
                results.key_influencers = convert_numpy(ki_result.to_dict())
                self.logger.success(f"Identified {len(ki_result.influencers)} key influencers")
            except Exception as e:
                self.logger.warning(f"Key influencers failed: {e}")
        
        # 3. DECOMPOSITION
        if primary_measure and dimensions:
            self.logger.info(f"Running decomposition for {primary_measure}...")
            try:
                from analysis.decomposition import decomposition_engine
                decomp = decomposition_engine.decompose(
                    df, 
                    measure_column=primary_measure,
                    dimension_columns=dimensions[:3],
                    max_depth=3
                )
                results.decomposition = convert_numpy(decomp.to_dict())
                self.logger.success("Generated decomposition tree")
            except Exception as e:
                self.logger.warning(f"Decomposition failed: {e}")
        
        # 4. OUTLIER DETECTION
        self.logger.info("Running outlier detection...")
        try:
            from analysis.outliers import outlier_detector
            for measure in measures[:5]:
                try:
                    outlier_result = outlier_detector.get_consensus_outliers(df, measure)
                    if outlier_result.outlier_count > 0:
                        results.outliers[measure] = convert_numpy(outlier_result.to_dict())
                except Exception:
                    pass
            self.logger.success(f"Analyzed outliers for {len(results.outliers)} measures")
        except Exception as e:
            self.logger.warning(f"Outlier detection failed: {e}")
        
        # 5. TREND ANALYSIS (if time column exists)
        if time_cols and measures:
            self.logger.info("Running trend analysis...")
            try:
                from analysis.trends import trend_detector
                for measure in measures[:3]:
                    try:
                        trend = trend_detector.detect_linear_trend(df, measure)
                        if trend.is_significant:
                            results.trends[measure] = convert_numpy(trend.to_dict())
                    except Exception:
                        pass
                self.logger.success(f"Found significant trends for {len(results.trends)} measures")
            except Exception as e:
                self.logger.warning(f"Trend analysis failed: {e}")
        
        # 6. DISTRIBUTION ANALYSIS
        self.logger.info("Running distribution analysis...")
        try:
            from analysis.statistical import statistical_analyzer
            for measure in measures[:5]:
                try:
                    dist = statistical_analyzer.analyze_distribution(df, measure)
                    results.distributions[measure] = convert_numpy(dist.to_dict())
                except Exception:
                    pass
            self.logger.success(f"Analyzed distributions for {len(results.distributions)} measures")
        except Exception as e:
            self.logger.warning(f"Distribution analysis failed: {e}")
        
        self.logger.success("=== ANALYSIS ORCHESTRATOR COMPLETE ===")
        return results


# Global instance
analysis_orchestrator = AnalysisOrchestrator()
