"""
Quick Insights Generator

Generates fast, automatic insights similar to Power BI Quick Insights.
Runs multiple analysis engines in parallel and ranks results.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from uuid import uuid4

import polars as pl

from api.schemas.responses import (
    Insight, InsightSeverity, InsightType, QuickInsightsResponse
)
from analysis.statistical import statistical_analyzer
from analysis.trends import trend_detector
from analysis.outliers import outlier_detector
from analysis.correlations import correlation_analyzer
from analysis.patterns import pattern_recognizer
from analysis.seasonality import seasonality_analyzer
from config import get_settings
from core.data_profiler import data_profiler
from insights.ranker import insight_ranker


class QuickInsightsGenerator:
    """
    Fast automatic insights generator.
    
    Runs multiple analysis engines in parallel to quickly surface
    the most interesting patterns in the data.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.executor = ThreadPoolExecutor(
            max_workers=self.settings.analysis.max_workers
        )
    
    async def generate(
        self,
        df: pl.DataFrame,
        session_id: str,
        top_n: Optional[int] = None,
    ) -> QuickInsightsResponse:
        """
        Generate quick insights for a dataset.
        
        Args:
            df: Polars DataFrame
            session_id: Session identifier
            top_n: Number of top insights to return
            
        Returns:
            QuickInsightsResponse with ranked insights
        """
        start_time = time.time()
        
        if top_n is None:
            top_n = self.settings.analysis.quick_insights_top_n
        
        # Get column info
        numeric_cols = data_profiler.get_numeric_columns(df)
        categorical_cols = data_profiler.get_categorical_columns(df)
        datetime_cols = data_profiler.get_datetime_columns(df)
        
        # Run analyses in parallel
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(self.executor, self._analyze_trends, df, numeric_cols),
            loop.run_in_executor(self.executor, self._analyze_outliers, df, numeric_cols),
            loop.run_in_executor(self.executor, self._analyze_correlations, df, numeric_cols),
            loop.run_in_executor(self.executor, self._analyze_patterns, df, categorical_cols),
            loop.run_in_executor(self.executor, self._analyze_distributions, df, numeric_cols),
        ]
        
        # Wait for all analyses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all insights
        all_insights = []
        for result in results:
            if isinstance(result, list):
                all_insights.extend(result)
        
        # Rank and diversify insights
        ranked = insight_ranker.rank_insights(all_insights)
        diversified = insight_ranker.diversify_insights(ranked)
        top_insights = diversified[:top_n]
        
        # Generate summary
        summary = self._generate_summary(df, top_insights)
        
        processing_time = (time.time() - start_time) * 1000
        
        return QuickInsightsResponse(
            session_id=session_id,
            generated_at=time.time(),
            processing_time_ms=processing_time,
            insights=top_insights,
            summary=summary,
        )
    
    def _analyze_trends(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
    ) -> list[Insight]:
        """Analyze trends in numeric columns."""
        insights = []
        
        for col in numeric_cols[:10]:  # Limit for speed
            try:
                trend = trend_detector.detect_linear_trend(df, col)
                
                if trend.is_significant and abs(trend.r_squared) > 0.3:
                    severity = insight_ranker.determine_severity(
                        p_value=trend.p_value,
                        magnitude=trend.r_squared,
                    )
                    
                    direction_icon = "📈" if trend.direction == "increasing" else "📉"
                    
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.TREND,
                        severity=severity,
                        title=f"{direction_icon} {col} shows {trend.strength} {trend.direction} trend",
                        description=f"The values in '{col}' show a statistically significant "
                                  f"{trend.direction} trend with {trend.percent_change:.1f}% change. "
                                  f"R² = {trend.r_squared:.2f}, indicating the trend explains "
                                  f"{trend.r_squared*100:.0f}% of the variance.",
                        score=trend.r_squared,
                        columns=[col],
                        metrics=trend.to_dict(),
                        chart_type="line",
                    ))
                    
                # Check for change points
                change_points = trend_detector.detect_change_points(df, col)
                if change_points:
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.ANOMALY,
                        severity=InsightSeverity.HIGH,
                        title=f"🔄 {len(change_points)} change point(s) detected in {col}",
                        description=f"Significant changes in the pattern of '{col}' were detected. "
                                  f"The most significant change occurred at index {change_points[0].index} "
                                  f"with a magnitude of {change_points[0].change_magnitude:.2f}.",
                        score=0.7,
                        columns=[col],
                        metrics={
                            "change_points": [cp.to_dict() for cp in change_points],
                            "count": len(change_points),
                        },
                        chart_type="line",
                    ))
            except Exception:
                continue
        
        return insights
    
    def _analyze_outliers(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
    ) -> list[Insight]:
        """Detect outliers in numeric columns."""
        insights = []
        
        for col in numeric_cols[:10]:
            try:
                # Use consensus outliers for reliability
                result = outlier_detector.get_consensus_outliers(df, col)
                
                if result.outlier_count > 0 and result.outlier_percentage > 0.5:
                    severity = insight_ranker.determine_severity(
                        outlier_percentage=result.outlier_percentage
                    )
                    
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.OUTLIER,
                        severity=severity,
                        title=f"⚠️ {result.outlier_count} outliers detected in {col}",
                        description=f"Found {result.outlier_count} outliers ({result.outlier_percentage:.1f}%) "
                                  f"in '{col}'. These values are significantly different from the typical "
                                  f"distribution and may warrant investigation.",
                        score=min(1.0, result.outlier_percentage / 10),
                        columns=[col],
                        metrics=result.to_dict(),
                        chart_type="box",
                    ))
            except Exception:
                continue
        
        return insights
    
    def _analyze_correlations(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
    ) -> list[Insight]:
        """Find significant correlations."""
        insights = []
        
        if len(numeric_cols) < 2:
            return insights
        
        try:
            correlations = correlation_analyzer.find_strongest_correlations(
                df, top_n=10, columns=numeric_cols[:15]
            )
            
            for corr in correlations[:5]:  # Top 5 correlations
                if abs(corr.pearson) < self.settings.analysis.min_correlation_strength:
                    continue
                
                severity = insight_ranker.determine_severity(
                    p_value=corr.pearson_pvalue,
                    magnitude=abs(corr.pearson),
                )
                
                direction_text = "positively" if corr.pearson > 0 else "negatively"
                icon = "🔗" if corr.pearson > 0 else "🔀"
                
                insights.append(Insight(
                    id=str(uuid4())[:8],
                    type=InsightType.CORRELATION,
                    severity=severity,
                    title=f"{icon} {corr.strength.capitalize()} {direction_text} correlation: {corr.column1} ↔ {corr.column2}",
                    description=f"'{corr.column1}' and '{corr.column2}' are {direction_text} correlated "
                              f"(r = {corr.pearson:.2f}). This {corr.strength} relationship is "
                              f"statistically significant (p < {corr.pearson_pvalue:.4f}).",
                    score=abs(corr.pearson),
                    columns=[corr.column1, corr.column2],
                    metrics=corr.to_dict(),
                    chart_type="scatter",
                ))
        except Exception:
            pass
        
        return insights
    
    def _analyze_patterns(
        self,
        df: pl.DataFrame,
        categorical_cols: list[str],
    ) -> list[Insight]:
        """Detect patterns in categorical columns."""
        insights = []
        
        for col in categorical_cols[:10]:
            try:
                # Majority detection
                majority = pattern_recognizer.detect_majority(df, col)
                
                if majority.is_supermajority:
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.PATTERN,
                        severity=InsightSeverity.MEDIUM,
                        title=f"🎯 Dominant value in {col}: '{majority.dominant_value}'",
                        description=f"The value '{majority.dominant_value}' appears in "
                                  f"{majority.dominant_percentage:.1f}% of all records for '{col}'. "
                                  f"This supermajority may indicate data concentration or a key category.",
                        score=majority.dominant_percentage / 100,
                        columns=[col],
                        metrics=majority.to_dict(),
                        chart_type="pie",
                    ))
                
                # Frequency pattern
                freq = pattern_recognizer.detect_frequency_pattern(df, col)
                
                if freq.distribution_type == "pareto":
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.PATTERN,
                        severity=InsightSeverity.MEDIUM,
                        title=f"📊 Pareto distribution in {col}",
                        description=f"'{col}' follows a Pareto (80/20) pattern where a small number "
                                  f"of values account for most of the occurrences. "
                                  f"Concentration ratio: {freq.concentration_ratio:.2f}.",
                        score=0.6,
                        columns=[col],
                        metrics=freq.to_dict(),
                        chart_type="bar",
                    ))
            except Exception:
                continue
        
        return insights
    
    def _analyze_distributions(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
    ) -> list[Insight]:
        """Analyze distribution shapes."""
        insights = []
        
        for col in numeric_cols[:10]:
            try:
                stats = statistical_analyzer.compute_descriptive_stats(df, col)
                dist = statistical_analyzer.analyze_distribution(df, col)
                
                # Check for high skewness
                if abs(stats.skewness) > 1:
                    direction = "right" if stats.skewness > 0 else "left"
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.DISTRIBUTION,
                        severity=InsightSeverity.LOW,
                        title=f"📐 {col} is highly skewed to the {direction}",
                        description=f"The distribution of '{col}' has a skewness of {stats.skewness:.2f}, "
                                  f"indicating a long tail on the {direction}. Consider log transformation "
                                  f"for analysis or median instead of mean.",
                        score=min(1.0, abs(stats.skewness) / 3),
                        columns=[col],
                        metrics={**stats.to_dict(), **dist.to_dict()},
                        chart_type="histogram",
                    ))
                
                # Check for bimodal distribution
                if dist.modality >= 2:
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.DISTRIBUTION,
                        severity=InsightSeverity.MEDIUM,
                        title=f"📊 Bimodal distribution detected in {col}",
                        description=f"'{col}' shows {dist.modality} distinct peaks in its distribution. "
                                  f"This may indicate distinct groups or subpopulations in the data.",
                        score=0.7,
                        columns=[col],
                        metrics=dist.to_dict(),
                        chart_type="histogram",
                    ))
            except Exception:
                continue
        
        return insights
    
    def _generate_summary(
        self,
        df: pl.DataFrame,
        insights: list[Insight],
    ) -> str:
        """Generate a text summary of the insights."""
        
        n_rows = len(df)
        n_cols = len(df.columns)
        n_insights = len(insights)
        
        if n_insights == 0:
            return f"Analyzed {n_rows:,} rows across {n_cols} columns. No significant patterns detected."
        
        # Count by type
        type_counts = {}
        for insight in insights:
            type_counts[insight.type] = type_counts.get(insight.type, 0) + 1
        
        type_summary = ", ".join(
            f"{count} {t.value}(s)" for t, count in type_counts.items()
        )
        
        # Get top insight
        top_insight = insights[0] if insights else None
        top_summary = f" Most significant: {top_insight.title}" if top_insight else ""
        
        return f"Analyzed {n_rows:,} rows × {n_cols} columns. Found {n_insights} insights: {type_summary}.{top_summary}"


# Global instance
quick_insights_generator = QuickInsightsGenerator()
