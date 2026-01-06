"""
Report Insights Generator

Generates comprehensive, in-depth report analysis.
More thorough than Quick Insights, suitable for detailed exploration.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from uuid import uuid4

import polars as pl

from api.schemas.responses import (
    Insight, InsightSeverity, InsightType, 
    ReportInsightsResponse, CorrelationInsight,
    KeyInfluencersResult as KeyInfluencersResponse,
    DecompositionResult as DecompositionResponse,
)
from analysis.statistical import statistical_analyzer
from analysis.trends import trend_detector
from analysis.outliers import outlier_detector
from analysis.correlations import correlation_analyzer
from analysis.key_influencers import key_influencers_analyzer
from analysis.decomposition import decomposition_engine
from analysis.patterns import pattern_recognizer
from analysis.seasonality import seasonality_analyzer
from config import get_settings
from core.data_profiler import data_profiler
from insights.ranker import insight_ranker


class ReportInsightsGenerator:
    """
    Comprehensive report insights generator.
    
    Provides in-depth analysis including key influencers,
    decomposition trees, and detailed statistical breakdowns.
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
        target_column: Optional[str] = None,
        measure_column: Optional[str] = None,
    ) -> ReportInsightsResponse:
        """
        Generate comprehensive report insights.
        
        Args:
            df: Polars DataFrame
            session_id: Session identifier
            target_column: Column for key influencers analysis
            measure_column: Column for decomposition analysis
            
        Returns:
            ReportInsightsResponse with detailed analysis
        """
        start_time = time.time()
        
        # Get data profile
        profile = data_profiler.profile(df)
        
        # Determine columns for advanced analysis
        numeric_cols = profile.numeric_columns
        categorical_cols = profile.categorical_columns
        
        # Auto-select target/measure if not specified
        if target_column is None and numeric_cols:
            target_column = numeric_cols[0]
        if measure_column is None and numeric_cols:
            measure_column = numeric_cols[0] if len(numeric_cols) == 1 else numeric_cols[-1]
        
        loop = asyncio.get_event_loop()
        
        # Run comprehensive analyses
        tasks = [
            loop.run_in_executor(self.executor, self._analyze_all_trends, df, numeric_cols),
            loop.run_in_executor(self.executor, self._analyze_all_outliers, df, numeric_cols),
            loop.run_in_executor(self.executor, self._analyze_all_distributions, df, numeric_cols),
            loop.run_in_executor(self.executor, self._analyze_all_patterns, df, categorical_cols),
            loop.run_in_executor(self.executor, self._analyze_seasonality, df, numeric_cols),
        ]
        
        # Add key influencers if we have a target
        key_influencers_result = None
        if target_column and len(df.columns) > 1:
            tasks.append(
                loop.run_in_executor(
                    self.executor, 
                    self._analyze_key_influencers, 
                    df, target_column
                )
            )
        
        # Add decomposition if we have categorical columns
        decomposition_result = None
        if measure_column and categorical_cols:
            tasks.append(
                loop.run_in_executor(
                    self.executor,
                    self._analyze_decomposition,
                    df, measure_column, categorical_cols
                )
            )
        
        # Wait for analyses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect insights
        all_insights = []
        for result in results[:5]:  # First 5 are basic insights
            if isinstance(result, list):
                all_insights.extend(result)
        
        # Extract key influencers and decomposition
        for result in results[5:]:
            if isinstance(result, dict):
                if "influencers" in result:
                    key_influencers_result = self._format_key_influencers(result)
                elif "root" in result:
                    decomposition_result = self._format_decomposition(result)
        
        # Get correlations
        correlations = self._get_correlation_insights(df, numeric_cols)
        
        # Rank insights
        ranked_insights = insight_ranker.rank_insights(all_insights)
        
        # Generate narrative
        narrative = self._generate_narrative(
            df, profile, ranked_insights, 
            correlations, key_influencers_result
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ReportInsightsResponse(
            session_id=session_id,
            generated_at=time.time(),
            processing_time_ms=processing_time,
            profile=profile,
            insights=ranked_insights[:20],  # Top 20 insights
            correlations=correlations[:10],
            key_influencers=key_influencers_result,
            decomposition=decomposition_result,
            narrative=narrative,
        )
    
    def _analyze_all_trends(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
    ) -> list[Insight]:
        """Comprehensive trend analysis for all columns."""
        insights = []
        
        for col in numeric_cols:
            try:
                # Linear trend
                trend = trend_detector.detect_linear_trend(df, col)
                
                if trend.is_significant:
                    insights.append(self._create_trend_insight(col, trend))
                
                # Mann-Kendall test for robustness
                mk_result = trend_detector.mann_kendall_test(df, col)
                
                if mk_result["significant"] and mk_result["trend"] != "no_trend":
                    if not trend.is_significant:
                        # Non-linear monotonic trend
                        insights.append(Insight(
                            id=str(uuid4())[:8],
                            type=InsightType.TREND,
                            severity=InsightSeverity.MEDIUM,
                            title=f"📈 Monotonic {mk_result['trend']} trend in {col}",
                            description=f"Mann-Kendall test detected a significant {mk_result['trend']} "
                                      f"monotonic trend in '{col}' (z = {mk_result['z_score']:.2f}).",
                            score=abs(mk_result["z_score"]) / 5,
                            columns=[col],
                            metrics=mk_result,
                            chart_type="line",
                        ))
                
                # Change point detection
                change_points = trend_detector.detect_change_points(df, col)
                if change_points:
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.ANOMALY,
                        severity=InsightSeverity.HIGH,
                        title=f"🔄 {len(change_points)} structural break(s) in {col}",
                        description=f"Detected significant changes in the pattern of '{col}'. "
                                  f"These change points may indicate regime shifts or events.",
                        score=0.8,
                        columns=[col],
                        metrics={"change_points": [cp.to_dict() for cp in change_points]},
                        chart_type="line",
                    ))
            except Exception:
                continue
        
        return insights
    
    def _create_trend_insight(self, col: str, trend) -> Insight:
        """Create insight from trend result."""
        severity = insight_ranker.determine_severity(
            p_value=trend.p_value,
            magnitude=trend.r_squared,
        )
        
        return Insight(
            id=str(uuid4())[:8],
            type=InsightType.TREND,
            severity=severity,
            title=f"📈 {trend.strength.capitalize()} {trend.direction} trend: {col}",
            description=f"'{col}' shows a {trend.strength} {trend.direction} trend with "
                      f"{trend.percent_change:.1f}% change (R² = {trend.r_squared:.3f}, "
                      f"p = {trend.p_value:.4f}).",
            score=trend.r_squared,
            columns=[col],
            metrics=trend.to_dict(),
            chart_type="line",
        )
    
    def _analyze_all_outliers(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
    ) -> list[Insight]:
        """Comprehensive outlier analysis."""
        insights = []
        
        for col in numeric_cols:
            try:
                # Multi-method consensus
                consensus = outlier_detector.get_consensus_outliers(df, col)
                
                if consensus.outlier_count > 0:
                    severity = insight_ranker.determine_severity(
                        outlier_percentage=consensus.outlier_percentage
                    )
                    
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.OUTLIER,
                        severity=severity,
                        title=f"⚠️ {consensus.outlier_count} consensus outliers in {col}",
                        description=f"Multiple outlier detection methods agree on "
                                  f"{consensus.outlier_count} outliers ({consensus.outlier_percentage:.1f}%) "
                                  f"in '{col}'. High-confidence anomalies worth investigating.",
                        score=min(1.0, consensus.outlier_percentage / 5),
                        columns=[col],
                        metrics=consensus.to_dict(),
                        chart_type="box",
                    ))
            except Exception:
                continue
        
        # Multivariate outliers
        if len(numeric_cols) >= 2:
            try:
                mv_result = outlier_detector.detect_isolation_forest(
                    df, numeric_cols[:10]
                )
                
                if mv_result.outlier_count > 0:
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.OUTLIER,
                        severity=InsightSeverity.HIGH,
                        title=f"🔍 {mv_result.outlier_count} multivariate outliers detected",
                        description=f"Isolation Forest detected {mv_result.outlier_count} rows that are "
                                  f"anomalous when considering multiple columns together. These may be "
                                  f"unusual combinations even if individual values seem normal.",
                        score=min(1.0, mv_result.outlier_percentage / 5),
                        columns=numeric_cols[:10],
                        metrics=mv_result.to_dict(),
                        chart_type="scatter",
                    ))
            except Exception:
                pass
        
        return insights
    
    def _analyze_all_distributions(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
    ) -> list[Insight]:
        """Analyze distribution shapes for all columns."""
        insights = []
        
        for col in numeric_cols:
            try:
                stats = statistical_analyzer.compute_descriptive_stats(df, col)
                dist = statistical_analyzer.analyze_distribution(df, col)
                
                # Coefficient of variation (variability)
                if stats.cv and stats.cv > 1:
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.DISTRIBUTION,
                        severity=InsightSeverity.MEDIUM,
                        title=f"📊 High variability in {col}",
                        description=f"'{col}' has a coefficient of variation of {stats.cv:.2f}, "
                                  f"indicating high relative variability. The standard deviation "
                                  f"is larger than the mean.",
                        score=min(1.0, stats.cv / 2),
                        columns=[col],
                        metrics=stats.to_dict(),
                        chart_type="histogram",
                    ))
                
                # Distribution type
                if dist.distribution_type == "bimodal":
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.DISTRIBUTION,
                        severity=InsightSeverity.MEDIUM,
                        title=f"📊 Bimodal distribution in {col}",
                        description=f"'{col}' shows two distinct peaks, suggesting two subpopulations "
                                  f"or distinct groups in the data.",
                        score=0.7,
                        columns=[col],
                        metrics=dist.to_dict(),
                        chart_type="histogram",
                    ))
            except Exception:
                continue
        
        return insights
    
    def _analyze_all_patterns(
        self,
        df: pl.DataFrame,
        categorical_cols: list[str],
    ) -> list[Insight]:
        """Analyze all categorical patterns."""
        insights = []
        
        for col in categorical_cols:
            try:
                majority = pattern_recognizer.detect_majority(df, col)
                freq = pattern_recognizer.detect_frequency_pattern(df, col)
                
                if majority.is_majority:
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.PATTERN,
                        severity=InsightSeverity.MEDIUM,
                        title=f"🎯 Dominant category in {col}",
                        description=f"'{majority.dominant_value}' represents "
                                  f"{majority.dominant_percentage:.1f}% of all values in '{col}'.",
                        score=majority.dominant_percentage / 100,
                        columns=[col],
                        metrics=majority.to_dict(),
                        chart_type="pie",
                    ))
                
                if freq.distribution_type in ("pareto", "zipf"):
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.PATTERN,
                        severity=InsightSeverity.LOW,
                        title=f"📊 {freq.distribution_type.capitalize()} distribution in {col}",
                        description=f"'{col}' follows a {freq.distribution_type} distribution "
                                  f"(entropy = {freq.entropy:.2f}, normalized = {freq.normalized_entropy:.2f}).",
                        score=0.5,
                        columns=[col],
                        metrics=freq.to_dict(),
                        chart_type="bar",
                    ))
            except Exception:
                continue
        
        return insights
    
    def _analyze_seasonality(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
    ) -> list[Insight]:
        """Detect seasonality patterns."""
        insights = []
        
        for col in numeric_cols[:5]:  # Limit for performance
            try:
                result = seasonality_analyzer.analyze(df, col)
                
                if result.has_seasonality and result.seasonal_components:
                    primary = result.seasonal_components[0]
                    
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type=InsightType.SEASONALITY,
                        severity=InsightSeverity.MEDIUM,
                        title=f"🔄 Seasonal pattern in {col}: {primary.interpretation}",
                        description=f"'{col}' shows a repeating pattern with period {primary.period} "
                                  f"(strength = {primary.strength:.2f}). {primary.interpretation}.",
                        score=primary.strength,
                        columns=[col],
                        metrics=result.to_dict(),
                        chart_type="line",
                    ))
            except Exception:
                continue
        
        return insights
    
    def _analyze_key_influencers(
        self,
        df: pl.DataFrame,
        target_column: str,
    ) -> dict:
        """Analyze key influencers for a target."""
        try:
            result = key_influencers_analyzer.analyze(df, target_column)
            return result.to_dict()
        except Exception:
            return {}
    
    def _analyze_decomposition(
        self,
        df: pl.DataFrame,
        measure_column: str,
        categorical_cols: list[str],
    ) -> dict:
        """Generate decomposition tree."""
        try:
            result = decomposition_engine.auto_decompose(df, measure_column)
            return result.to_dict()
        except Exception:
            return {}
    
    def _get_correlation_insights(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
    ) -> list[CorrelationInsight]:
        """Get correlation insights."""
        if len(numeric_cols) < 2:
            return []
        
        try:
            pairs = correlation_analyzer.find_strongest_correlations(
                df, top_n=10, columns=numeric_cols[:15]
            )
            
            return [
                CorrelationInsight(
                    id=str(uuid4())[:8],
                    type=InsightType.CORRELATION,
                    severity=insight_ranker.determine_severity(
                        p_value=p.pearson_pvalue,
                        magnitude=abs(p.pearson),
                    ),
                    title=f"{p.column1} ↔ {p.column2}",
                    description=f"Correlation: {p.pearson:.3f}",
                    score=abs(p.pearson),
                    columns=[p.column1, p.column2],
                    metrics=p.to_dict(),
                    column1=p.column1,
                    column2=p.column2,
                    correlation=p.pearson,
                    p_value=p.pearson_pvalue,
                    method="pearson",
                )
                for p in pairs
            ]
        except Exception:
            return []
    
    def _format_key_influencers(self, result: dict) -> Optional[Any]:
        """Format key influencers result."""
        if not result or "influencers" not in result:
            return None
        return result
    
    def _format_decomposition(self, result: dict) -> Optional[Any]:
        """Format decomposition result."""
        if not result or "root" not in result:
            return None
        return result
    
    def _generate_narrative(
        self,
        df: pl.DataFrame,
        profile,
        insights: list[Insight],
        correlations: list,
        key_influencers,
    ) -> str:
        """Generate a narrative summary of the report."""
        
        parts = [
            f"## Data Overview\n",
            f"Analyzed **{profile.row_count:,}** rows and **{profile.column_count}** columns.\n",
            f"- Numeric columns: {len(profile.numeric_columns)}\n",
            f"- Categorical columns: {len(profile.categorical_columns)}\n",
            f"- Memory usage: {profile.memory_usage_mb:.2f} MB\n\n",
        ]
        
        # Key findings
        if insights:
            parts.append("## Key Findings\n")
            for insight in insights[:5]:
                parts.append(f"- **{insight.title}**: {insight.description[:200]}...\n")
            parts.append("\n")
        
        # Correlations
        if correlations:
            parts.append("## Notable Correlations\n")
            for corr in correlations[:3]:
                parts.append(f"- {corr.title} (r = {corr.correlation:.3f})\n")
            parts.append("\n")
        
        # Key influencers
        if key_influencers and key_influencers.get("influencers"):
            top_inf = key_influencers["influencers"][0]
            parts.append(
                f"## Key Influencers\n"
                f"Top influencer for '{key_influencers['target_column']}': "
                f"**{top_inf['feature']}** (importance: {top_inf['importance']:.3f})\n\n"
            )
        
        return "".join(parts)


# Global instance
report_insights_generator = ReportInsightsGenerator()
