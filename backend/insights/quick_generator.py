"""
Quick Insights Generator

Generates fast, automatic insights similar to Power BI Quick Insights.
Uses data understanding to intelligently select appropriate analyses.
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
from core.data_understanding import data_understanding_engine, AnalysisScope, DataUnderstanding
from insights.ranker import insight_ranker
from core.logging_config import insights_logger as logger


class QuickInsightsGenerator:
    """
    Fast automatic insights generator.
    
    First understands the data structure, then runs appropriate
    analyses based on data semantics (not blindly running all algorithms).
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
        include_advanced: bool = False,
    ) -> QuickInsightsResponse:
        """
        Generate quick insights for a dataset.
        
        1. First understands the data structure and semantics
        2. Determines which analyses are appropriate
        3. Runs analyses (globally or by group as needed)
        4. Ranks and returns top insights
        
        If include_advanced=True, also generates:
        - Full data profile
        - All correlations
        - Key influencers analysis
        - Decomposition tree
        """
        start_time = time.time()
        logger.info(f"=== STARTING QUICK INSIGHTS GENERATION ===")
        logger.info(f"Session: {session_id}")
        logger.info(f"Data shape: {len(df):,} rows x {len(df.columns)} columns")
        
        if top_n is None:
            top_n = self.settings.analysis.quick_insights_top_n
        logger.info(f"Max insights to return: {top_n}")
        
        # STEP 1: Understand the data
        logger.info("STEP 1: Understanding data structure...")
        understanding = data_understanding_engine.understand(df)
        logger.success(f"Data type: {understanding.dataset_type.value}")
        logger.success(f"Business domain: {understanding.business_domain}")
        
        # Get column classifications
        measure_cols = [m.name for m in understanding.measure_columns]
        grouping_cols = [g.name for g in understanding.grouping_columns]
        time_cols = [t.name for t in understanding.time_columns]
        
        logger.info(f"Measures: {measure_cols}")
        logger.info(f"Dimensions: {grouping_cols}")
        logger.info(f"Time columns: {time_cols}")
        
        # Get recommended analyses
        recommendations = {
            r.analysis_name: r 
            for r in understanding.analysis_recommendations 
            if r.applicable
        }
        logger.info(f"Applicable analyses: {list(recommendations.keys())}")
        
        loop = asyncio.get_event_loop()
        all_insights = []
        
        # ============================================================
        # NEW ANALYSIS-FIRST ARCHITECTURE
        # ============================================================
        
        # STEP 2: Run ALL analysis upfront
        logger.info("STEP 2: Running comprehensive analysis (analysis-first)...")
        from analysis.orchestrator import analysis_orchestrator
        analysis_results = await analysis_orchestrator.run_all(df, understanding)
        logger.success("All analysis complete")
        
        # STEP 3: Generate insights FROM analysis results
        logger.info("STEP 3: Generating insights FROM analysis results...")
        from insights.from_analysis import insight_generator_from_analysis, AnalysisDrivenInsight
        from api.schemas.responses import PowerBIInsightModel, DatasetSummary
        
        analysis_insights = insight_generator_from_analysis.generate_all(
            analysis_results, understanding, max_insights=top_n
        )
        logger.success(f"Generated {len(analysis_insights)} analysis-driven insights")
        
        # STEP 4: Also generate Power BI style comparison insights for user-friendly output
        logger.info("STEP 4: Adding comparison insights...")
        from insights.powerbi_style import PowerBIInsightsGenerator
        
        pbi_generator = PowerBIInsightsGenerator()
        comparison_insights = pbi_generator.generate_all(
            df, understanding, max_insights=top_n // 2  # Half from comparisons
        )
        logger.success(f"Generated {len(comparison_insights)} comparison insights")
        
        # STEP 5: Merge and deduplicate insights
        logger.info("STEP 5: Merging analysis insights with comparison insights...")
        
        # Convert analysis insights to PowerBIInsightModel
        analysis_models = [
            PowerBIInsightModel(
                insight_type=i.insight_type,
                measure=i.measure,
                dimension=i.dimension,
                statement=i.statement,
                chart_type=i.chart_type,
                chart_data=i.chart_data,
                score=i.score,
            )
            for i in analysis_insights
        ]
        
        # Convert comparison insights to PowerBIInsightModel  
        comparison_models = [
            PowerBIInsightModel(
                insight_type=i.insight_type,
                measure=i.measure,
                dimension=i.dimension,
                statement=i.statement,
                chart_type=i.chart_type,
                chart_data=i.chart_data,
                score=i.score,
            )
            for i in comparison_insights
        ]
        
        # Merge: analysis insights first (they're backed by stats), then comparisons
        all_models = analysis_models + comparison_models
        
        # Sort by score and take top N
        all_models.sort(key=lambda x: x.score, reverse=True)
        insights_models = all_models[:top_n]
        
        logger.success(f"Final merged insights: {len(insights_models)}")
        
        # STEP 6: Generate narrative (optional LLM enhancement)
        logger.info("STEP 6: Generating narrative summary with LLM...")
        from insights.narrative_generator import narrative_generator
        
        # Create dummy insights for narrative (using the models we have)
        dummy_insights = []  # narrative generator uses different format
        
        try:
            # Pass analysis insights to narrative generator
            narrative = await narrative_generator.generate_executive_summary(
                df, dummy_insights, understanding
            )
            logger.success(f"LLM narrative generated ({len(narrative) if narrative else 0} chars)")
        except Exception as e:
            logger.warning(f"LLM narrative failed: {e}")
            narrative = None  # LLM not available, that's fine
        
        # Generate recommendations from key influencers
        logger.info("Generating recommendations...")
        recommendations = self._generate_smart_recommendations(analysis_results, understanding)
        logger.success(f"Generated {len(recommendations)} recommendations")
        
        # Build dataset summary with semantic understanding
        logger.info("Building final response...")
        data_summary = DatasetSummary(
            row_count=len(df),
            column_count=len(df.columns),
            dataset_type=understanding.dataset_type.value,
            primary_grouper=understanding.primary_grouper,
            measures=[m.name for m in understanding.measure_columns],
            dimensions=[g.name for g in understanding.grouping_columns[:5]],
            # Semantic understanding
            business_domain=understanding.business_domain,
            data_story=understanding.data_story,
            key_questions=understanding.key_questions,
        )
        
        # Convert story columns to API models
        from api.schemas.responses import StoryColumnModel
        story_columns = [
            StoryColumnModel(
                name=sc.name,
                role=sc.role,
                business_meaning=sc.business_meaning,
                importance=sc.importance,
            )
            for sc in understanding.story_columns
        ]
        
        # ADVANCED ANALYSIS - now uses results already computed by analysis_orchestrator
        profile = None
        
        if include_advanced:
            logger.info("STEP 7: Including full data profile...")
            
            # Full data profile
            try:
                profile = data_profiler.profile(df)
                logger.success("Generated data profile")
            except Exception as e:
                logger.warning(f"Profile failed: {e}")
        
        # Analysis results are already computed - just extract them
        correlations = analysis_results.correlations
        key_influencers_result = analysis_results.key_influencers
        decomposition_result = analysis_results.decomposition
        
        processing_time = (time.time() - start_time) * 1000
        logger.success(f"=== QUICK INSIGHTS COMPLETE in {processing_time:.0f}ms ===")
        logger.info(f"Returning {len(insights_models)} insights, {len(recommendations)} recommendations")
        
        return QuickInsightsResponse(
            session_id=session_id,
            generated_at=time.time(),
            processing_time_ms=processing_time,
            data_summary=data_summary,
            story_columns=story_columns,
            insights=insights_models,
            recommendations=recommendations,
            narrative=narrative,
            # Advanced analysis (only if include_advanced=True)
            profile=profile,
            correlations=correlations,
            key_influencers=key_influencers_result,
            decomposition=decomposition_result,
        )
    
    def _create_understanding_insight(self, understanding) -> Insight:
        """Create an insight about the data structure itself."""
        
        group_text = ""
        if understanding.primary_grouper:
            group_text = f" Data should be analyzed by '{understanding.primary_grouper}'."
        
        time_text = ""
        if understanding.time_columns:
            time_text = f" Time column: '{understanding.time_columns[0].name}'."
        
        return Insight(
            id=str(uuid4())[:8],
            type=InsightType.SUMMARY,
            severity=InsightSeverity.MEDIUM,
            title=f"📋 Dataset identified as {understanding.dataset_type.value.replace('_', ' ')}",
            description=f"This dataset has {understanding.row_count:,} rows and "
                      f"{len(understanding.measure_columns)} measure columns.{group_text}{time_text} "
                      f"Analysis scope: {understanding.recommended_scope.value}.",
            score=0.9,
            columns=[],
            metrics=understanding.to_dict(),
        )
    
    def _generate_smart_recommendations(
        self, 
        analysis_results,
        understanding: DataUnderstanding,
    ) -> list[str]:
        """
        Generate smart recommendations based on analysis results.
        
        Uses key influencers, correlations, and decomposition to create
        actionable business recommendations.
        """
        from analysis.orchestrator import AnalysisResults
        
        recs = []
        
        # From key influencers
        if analysis_results.key_influencers:
            influencers = analysis_results.key_influencers.get("influencers", [])
            if influencers:
                top_inf = influencers[0]
                feature = top_inf.get("feature", "Unknown")
                target = analysis_results.key_influencers.get("target_column", "the measure")
                recs.append(f"Focus on '{feature}' - it's the top driver of {target}")
        
        # From strong correlations
        if analysis_results.correlations:
            for corr in analysis_results.correlations[:2]:
                r = corr.get("pearson", corr.get("correlation", 0))
                if abs(r) > 0.7:
                    col1 = corr.get("column1", "X")
                    col2 = corr.get("column2", "Y")
                    if r > 0:
                        recs.append(f"Consider '{col1}' and '{col2}' together - they're strongly linked")
                    else:
                        recs.append(f"Monitor tradeoff between '{col1}' and '{col2}' - they move opposite")
        
        # From decomposition
        if analysis_results.decomposition:
            breakdown = analysis_results.decomposition.get("breakdown", {})
            for dim, data in breakdown.items():
                if isinstance(data, dict):
                    contribs = data.get("contributions", [])
                    if contribs and contribs[0].get("percentage", 0) > 50:
                        top_val = contribs[0].get("value", "Unknown")
                        pct = contribs[0].get("percentage", 0)
                        recs.append(f"'{top_val}' dominates {dim} at {pct:.0f}% - consider segment-specific strategies")
                        break
        
        # From outliers
        if analysis_results.outliers:
            outlier_measures = list(analysis_results.outliers.keys())
            if outlier_measures:
                recs.append(f"Review outliers in {', '.join(outlier_measures[:2])} - they may be errors or opportunities")
        
        # From trends
        if analysis_results.trends:
            for measure, trend in analysis_results.trends.items():
                direction = trend.get("direction", "stable")
                change = trend.get("change_percent", 0)
                if direction == "increasing" and change > 10:
                    recs.append(f"Capitalize on rising {measure} (+{change:.0f}%)")
                elif direction == "decreasing" and change < -10:
                    recs.append(f"Investigate declining {measure} ({change:.0f}%)")
        
        # Default if none
        if not recs:
            if understanding.primary_grouper:
                recs.append(f"Analyze performance by '{understanding.primary_grouper}' to find opportunities")
            recs.append("Continue monitoring key metrics for emerging patterns")
        
        return recs[:5]  # Top 5 recommendations
    
    def _analyze_trends_by_group(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
        group_by: str,
    ) -> list[Insight]:
        """Analyze trends within each group separately."""
        insights = []
        
        # Get unique groups, excluding nulls
        groups = df[group_by].drop_nulls().unique().to_list()[:10]
        
        for group_val in groups:
            if group_val is None:
                continue
            group_df = df.filter(pl.col(group_by) == group_val)
            
            if len(group_df) < 10:  # Skip small groups
                continue
            
            for col in numeric_cols[:5]:  # Limit columns per group
                try:
                    trend = trend_detector.detect_linear_trend(group_df, col)
                    
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
                            title=f"{direction_icon} {col} trend in {group_by}='{group_val}'",
                            description=f"For {group_by}='{group_val}', '{col}' shows a "
                                      f"{trend.direction} trend ({trend.percent_change:.1f}% change, R²={trend.r_squared:.2f}).",
                            score=trend.r_squared,
                            columns=[col, group_by],
                            metrics={**trend.to_dict(), "group": str(group_val)},
                            chart_type="line",
                        ))
                except Exception:
                    continue
        
        return insights
    
    def _analyze_outliers_by_group(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
        group_by: str,
    ) -> list[Insight]:
        """Detect outliers within each group."""
        insights = []
        
        # Get unique groups, excluding nulls
        groups = df[group_by].drop_nulls().unique().to_list()[:10]
        
        for group_val in groups:
            if group_val is None:
                continue
            group_df = df.filter(pl.col(group_by) == group_val)
            
            if len(group_df) < 10:
                continue
            
            for col in numeric_cols[:5]:
                try:
                    result = outlier_detector.get_consensus_outliers(group_df, col)
                    
                    if result.outlier_count > 0 and result.outlier_percentage > 1:
                        insights.append(Insight(
                            id=str(uuid4())[:8],
                            type=InsightType.OUTLIER,
                            severity=insight_ranker.determine_severity(
                                outlier_percentage=result.outlier_percentage
                            ),
                            title=f"⚠️ {result.outlier_count} outliers in {col} for {group_by}='{group_val}'",
                            description=f"Within {group_by}='{group_val}', found {result.outlier_count} "
                                      f"outliers ({result.outlier_percentage:.1f}%) in '{col}'.",
                            score=min(1.0, result.outlier_percentage / 5),
                            columns=[col, group_by],
                            metrics={**result.to_dict(), "group": str(group_val)},
                            chart_type="box",
                        ))
                except Exception:
                    continue
        
        return insights
    
    def _analyze_group_comparison(
        self,
        df: pl.DataFrame,
        measure_cols: list[str],
        group_by: str,
    ) -> list[Insight]:
        """Compare measures across groups."""
        insights = []
        
        if not group_by or group_by not in df.columns:
            return insights
        
        for col in measure_cols[:5]:
            try:
                # Calculate stats per group
                group_stats = (
                    df.group_by(group_by)
                    .agg([
                        pl.col(col).mean().alias("mean"),
                        pl.col(col).std().alias("std"),
                        pl.count().alias("count"),
                    ])
                    .sort("mean", descending=True)
                )
                
                if len(group_stats) < 2:
                    continue
                
                # Find biggest difference
                means = group_stats["mean"].to_list()
                groups = group_stats[group_by].to_list()
                
                max_val = max(means)
                min_val = min(means)
                
                if min_val > 0:
                    ratio = max_val / min_val
                    if ratio > 1.5:  # At least 50% difference
                        max_group = groups[means.index(max_val)]
                        min_group = groups[means.index(min_val)]
                        
                        insights.append(Insight(
                            id=str(uuid4())[:8],
                            type=InsightType.PATTERN,
                            severity=InsightSeverity.HIGH,
                            title=f"📊 {col} varies {ratio:.1f}x across {group_by}",
                            description=f"'{col}' averages {max_val:.2f} for '{max_group}' vs "
                                      f"{min_val:.2f} for '{min_group}' - a {ratio:.1f}x difference.",
                            score=min(1.0, ratio / 3),
                            columns=[col, group_by],
                            metrics={
                                "group_column": group_by,
                                "highest_group": str(max_group),
                                "highest_value": float(max_val),
                                "lowest_group": str(min_group),
                                "lowest_value": float(min_val),
                                "ratio": float(ratio),
                            },
                            chart_type="bar",
                        ))
            except Exception:
                continue
        
        return insights
    
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
    
    def _generate_smart_summary(
        self,
        df: pl.DataFrame,
        insights: list[Insight],
        understanding,
    ) -> str:
        """Generate a context-aware summary of the insights."""
        
        n_rows = len(df)
        n_cols = len(df.columns)
        n_insights = len(insights)
        
        # Dataset type context
        dtype_text = understanding.dataset_type.value.replace("_", " ")
        
        if n_insights == 0:
            return f"Analyzed {n_rows:,} rows ({dtype_text} data). No significant patterns detected."
        
        # Context about grouping
        group_context = ""
        if understanding.primary_grouper:
            n_groups = df[understanding.primary_grouper].n_unique()
            group_context = f" Analyzed across {n_groups} {understanding.primary_grouper} groups."
        
        # Count by type
        type_counts = {}
        for insight in insights:
            type_counts[insight.type] = type_counts.get(insight.type, 0) + 1
        
        type_summary = ", ".join(
            f"{count} {t.value}(s)" for t, count in type_counts.items()
        )
        
        # Get top insight (skip the summary insight)
        top_insight = None
        for insight in insights:
            if insight.type != InsightType.SUMMARY:
                top_insight = insight
                break
        
        top_summary = f" Most significant: {top_insight.title}" if top_insight else ""
        
        return f"{dtype_text.capitalize()} dataset with {n_rows:,} rows.{group_context} Found {n_insights} insights: {type_summary}.{top_summary}"
    
    def _generate_summary(
        self,
        df: pl.DataFrame,
        insights: list[Insight],
    ) -> str:
        """Fallback summary method."""
        n_rows = len(df)
        n_cols = len(df.columns)
        n_insights = len(insights)
        
        if n_insights == 0:
            return f"Analyzed {n_rows:,} rows across {n_cols} columns. No significant patterns detected."
        
        return f"Analyzed {n_rows:,} rows × {n_cols} columns. Found {n_insights} insights."


# Global instance
quick_insights_generator = QuickInsightsGenerator()
