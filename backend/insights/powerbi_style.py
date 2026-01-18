"""
Power BI Style Insights Generator

Generates all 10 Power BI Quick Insights types:
1. Category outliers (top/bottom) - "high_value"
2. Change points in time series - "change_point"
3. Correlation - "correlation"
4. Low Variance - "low_variance"
5. Majority (Major factors) - "majority"
6. Outliers - "outlier"
7. Overall trends in time series - "trend"
8. Seasonality in time series - "seasonality"
9. Steady share - "steady_share"
10. Time series outliers - "time_series_outlier"
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from api.schemas.responses import Insight, InsightSeverity, InsightType
from core.data_understanding import data_understanding_engine, DataUnderstanding


@dataclass
class PowerBIInsight:
    """Power BI style insight with chart data."""
    
    insight_type: str
    measure: str
    dimension: Optional[str]
    statement: str
    chart_type: str
    chart_data: dict[str, Any]
    score: float = 0.5
    
    def to_insight(self) -> Insight:
        """Convert to API Insight model."""
        type_map = {
            "majority": InsightType.PATTERN,
            "trend": InsightType.TREND,
            "outlier": InsightType.OUTLIER,
            "correlation": InsightType.CORRELATION,
            "low_variance": InsightType.DISTRIBUTION,
            "high_value": InsightType.PATTERN,
            "change_point": InsightType.TREND,
            "seasonality": InsightType.SEASONALITY,
            "steady_share": InsightType.PATTERN,
            "time_series_outlier": InsightType.OUTLIER,
        }
        
        return Insight(
            id=str(uuid4())[:8],
            type=type_map.get(self.insight_type, InsightType.PATTERN),
            severity=InsightSeverity.MEDIUM,
            title=f"{self.measure} BY {self.dimension.upper() if self.dimension else 'VALUE'}",
            description=self.statement,
            score=self.score,
            columns=[self.measure] + ([self.dimension] if self.dimension else []),
            metrics={
                "insight_type": self.insight_type,
                "measure": self.measure,
                "dimension": self.dimension,
            },
            chart_type=self.chart_type,
            chart_data=self.chart_data,
        )


class PowerBIInsightsGenerator:
    """
    Generates Power BI Quick Insights that answer key business questions.
    
    Approach:
    1. Generate ALL possible insights (many)
    2. Filter and deduplicate
    3. Use LLM to enhance top insights (make them contextual)
    """
    
    def __init__(self):
        from core.logging_config import insights_logger
        self.logger = insights_logger
    
    def generate_all(
        self,
        df: pl.DataFrame,
        understanding: DataUnderstanding,
        max_insights: int = 20,
    ) -> list[PowerBIInsight]:
        """
        Generate insights in 3 phases:
        1. Generate ALL possible insights (many)
        2. Deduplicate and score
        3. Return top N
        """
        self.logger.info(f"Starting insight generation for {len(df):,} rows, {len(df.columns)} columns")
        all_insights = []
        
        # Get story columns - focus on what matters most
        story_cols = understanding.story_columns
        self.logger.debug(f"Story columns identified: {[sc.name for sc in story_cols]}")
        
        # Extract primary measure and key dimensions from story columns
        primary_measure = None
        secondary_measures = []
        key_dimensions = []
        
        for sc in story_cols:
            if sc.role == "primary_measure":
                primary_measure = sc.name
            elif sc.role == "secondary_measure":
                secondary_measures.append(sc.name)
            elif sc.role == "key_dimension":
                key_dimensions.append(sc.name)
        
        # Fallback to all measures/dimensions
        all_measures = [m.name for m in understanding.measure_columns]
        all_dimensions = [g.name for g in understanding.grouping_columns]
        
        if not primary_measure and all_measures:
            primary_measure = all_measures[0]
        if not key_dimensions and all_dimensions:
            key_dimensions = all_dimensions[:5]  # Consider more dimensions
        
        self.logger.info(f"Primary measure: {primary_measure}")
        self.logger.info(f"Secondary measures: {secondary_measures}")
        self.logger.info(f"Key dimensions: {key_dimensions}")
        self.logger.info(f"All measures: {all_measures}")
        self.logger.info(f"All dimensions: {all_dimensions}")
        
        # ========== PHASE 1: GENERATE ALL POSSIBLE INSIGHTS ==========
        
        # 1. Comparison insights (highest/lowest) - for each measure x dimension
        self.logger.info("Generating comparison insights...")
        for measure in all_measures[:5]:
            for dim in all_dimensions[:5]:
                insight = self._answer_highest_lowest(df, measure, dim)
                if insight:
                    all_insights.append(insight)
        self.logger.success(f"Generated {len([i for i in all_insights if i.insight_type == 'comparison'])} comparison insights")
        
        # 2. Distribution insights - for each measure
        self.logger.info("Generating distribution insights...")
        for measure in all_measures[:5]:
            insight = self._analyze_distribution(df, measure)
            if insight:
                all_insights.append(insight)
        self.logger.success(f"Generated {len([i for i in all_insights if i.insight_type == 'distribution'])} distribution insights")
        
        # 3. Relationship insights - between all measure pairs
        self.logger.info("Generating relationship insights...")
        for i, m1 in enumerate(all_measures[:5]):
            for m2 in all_measures[i+1:5]:
                insight = self._analyze_relationship(df, m1, m2)
                if insight:
                    all_insights.append(insight)
        self.logger.success(f"Generated {len([i for i in all_insights if i.insight_type == 'relationship'])} relationship insights")
        
        # 4. High value insights
        self.logger.info("Generating high value insights...")
        high_value_insights = self._find_high_value_insights(df, all_measures[:5], all_dimensions[:5])
        all_insights.extend(high_value_insights)
        self.logger.success(f"Generated {len(high_value_insights)} high value insights")
        
        # 5. Majority insights
        self.logger.info("Generating majority insights...")
        majority_insights = self._find_majority_insights(df, all_measures[:5], all_dimensions[:5])
        all_insights.extend(majority_insights)
        self.logger.success(f"Generated {len(majority_insights)} majority insights")
        
        # 6. Low variance insights
        self.logger.info("Generating low variance insights...")
        low_var_insights = self._find_low_variance_insights(df, all_measures[:5], all_dimensions[:5])
        all_insights.extend(low_var_insights)
        self.logger.success(f"Generated {len(low_var_insights)} low variance insights")
        
        # 7. Outlier insights
        self.logger.info("Generating outlier insights...")
        outlier_insights = self._find_outlier_insights(df, all_measures[:5], all_dimensions[:5])
        all_insights.extend(outlier_insights)
        self.logger.success(f"Generated {len(outlier_insights)} outlier insights")
        
        # 8. Correlation insights
        self.logger.info("Generating correlation insights...")
        corr_insights = self._find_correlation_insights(df, all_measures[:5])
        all_insights.extend(corr_insights)
        self.logger.success(f"Generated {len(corr_insights)} correlation insights")
        
        # 9-10. Time-based insights (if applicable)
        from core.data_understanding import DatasetType
        is_time_series = understanding.dataset_type in (
            DatasetType.TIME_SERIES, 
            DatasetType.PANEL,
            DatasetType.TRANSACTIONAL,
        )
        
        if is_time_series and understanding.time_columns:
            self.logger.info("Generating time-based insights...")
            trend_insights = self._find_trend_insights(df, all_measures[:3], understanding)
            all_insights.extend(trend_insights)
            self.logger.success(f"Generated {len(trend_insights)} trend insights")
        else:
            self.logger.warning("No time columns detected - skipping time-based insights")
        
        self.logger.info(f"Total raw insights generated: {len(all_insights)}")
        
        # ========== PHASE 2: DEDUPLICATE AND FILTER ==========
        
        self.logger.info("Deduplicating insights...")
        unique_insights = self._deduplicate_insights(all_insights)
        self.logger.success(f"After deduplication: {len(unique_insights)} insights")
        
        # ========== PHASE 3: SORT AND RETURN TOP N ==========
        
        unique_insights.sort(key=lambda x: x.score, reverse=True)
        final_insights = unique_insights[:max_insights]
        
        self.logger.success(f"Returning top {len(final_insights)} insights")
        for i, ins in enumerate(final_insights[:5]):
            self.logger.debug(f"  #{i+1}: {ins.insight_type} - {ins.statement[:60]}...")
        
        return final_insights
    
    # ========== QUESTION-ANSWERING METHODS ==========
    
    def _answer_highest_lowest(
        self,
        df: pl.DataFrame,
        measure: str,
        dimension: str,
    ) -> Optional[PowerBIInsight]:
        """
        Answer: Which [dimension] has highest/lowest [measure]?
        
        Smartly chooses aggregation:
        - SUM for quantities, counts, sales totals (additive measures)
        - AVERAGE for prices, rates, costs per unit (non-additive measures)
        """
        try:
            measure_lower = measure.lower()
            
            # Determine if this is a "rate" measure (should use average) or "total" measure (should use sum)
            rate_keywords = ["price", "cost", "rate", "per", "average", "avg", "margin", "percent", "%"]
            is_rate_measure = any(kw in measure_lower for kw in rate_keywords)
            
            # Choose aggregation
            if is_rate_measure:
                agg_func = pl.col(measure).mean().alias("agg_value")
                agg_label = "average"
            else:
                agg_func = pl.col(measure).sum().alias("agg_value")
                agg_label = "total"
            
            agg = (
                df.group_by(dimension)
                .agg(agg_func)
                .sort("agg_value", descending=True)
            )
            
            if len(agg) < 2:
                return None
            
            labels = agg[dimension].to_list()
            values = agg["agg_value"].to_list()
            
            if max(values) == min(values):
                return None
            
            highest, lowest = labels[0], labels[-1]
            highest_val, lowest_val = values[0], values[-1]
            
            diff_pct = ((highest_val - lowest_val) / abs(lowest_val) * 100) if lowest_val != 0 else 0
            
            def fmt(v):
                if abs(v) >= 1_000_000:
                    return f"{v/1_000_000:.1f}M"
                elif abs(v) >= 1_000:
                    return f"{v/1_000:.1f}K"
                elif abs(v) >= 1:
                    return f"{v:.2f}"
                return f"{v:.4f}"
            
            # Build contextual statement
            if is_rate_measure:
                statement = f"'{highest}' has the highest average {measure} (${fmt(highest_val)}), {diff_pct:.0f}% higher than '{lowest}' (${fmt(lowest_val)})."
            else:
                statement = f"'{highest}' has the highest {agg_label} {measure} ({fmt(highest_val)}), {diff_pct:.0f}% more than '{lowest}' ({fmt(lowest_val)})."
            
            # Try to add context: Which item type is driving this?
            context_dimension = None
            for ctx_dim in ["Item Type", "Product", "Category", "Product Name", "ITEM TYPE"]:
                if ctx_dim in df.columns and ctx_dim != dimension:
                    context_dimension = ctx_dim
                    break
            
            if context_dimension:
                try:
                    # Find top contributor within the highest-performing dimension value
                    top_subset = df.filter(pl.col(dimension) == highest)
                    if is_rate_measure:
                        top_ctx = (
                            top_subset.group_by(context_dimension)
                            .agg(pl.col(measure).mean().alias("ctx_val"))
                            .sort("ctx_val", descending=True)
                            .head(1)
                        )
                    else:
                        top_ctx = (
                            top_subset.group_by(context_dimension)
                            .agg(pl.col(measure).sum().alias("ctx_val"))
                            .sort("ctx_val", descending=True)
                            .head(1)
                        )
                    
                    if len(top_ctx) > 0:
                        top_item = top_ctx[context_dimension][0]
                        statement += f" Top {context_dimension.lower()}: '{top_item}'."
                except:
                    pass
            
            return PowerBIInsight(
                insight_type="comparison",
                measure=measure,
                dimension=dimension,
                statement=statement,
                chart_type="bar",
                chart_data={
                    "labels": [str(l) for l in labels[:10]],
                    "values": [float(v) for v in values[:10]],
                    "highest": str(highest),
                    "lowest": str(lowest),
                    "difference_percent": round(diff_pct, 1),
                    "aggregation": agg_label,
                },
                score=min(1.0, 0.8 + diff_pct / 200),
            )
        except Exception as e:
            self.logger.warning(f"Failed to generate comparison insight: {e}")
            return None
    
    def _analyze_distribution(
        self,
        df: pl.DataFrame,
        measure: str,
    ) -> Optional[PowerBIInsight]:
        """Answer: What is the distribution of [measure]?"""
        try:
            values = df[measure].drop_nulls().to_numpy()
            if len(values) < 10:
                return None
            
            mean_val = np.mean(values)
            median_val = np.median(values)
            min_val, max_val = np.min(values), np.max(values)
            
            if mean_val > median_val * 1.1:
                skew = "right-skewed (long tail of high values)"
            elif median_val > mean_val * 1.1:
                skew = "left-skewed (long tail of low values)"
            else:
                skew = "roughly symmetric"
            
            def fmt(v):
                if abs(v) >= 1_000_000:
                    return f"{v/1_000_000:.1f}M"
                elif abs(v) >= 1_000:
                    return f"{v/1_000:.1f}K"
                return f"{v:.0f}"
            
            statement = f"{measure} ranges from {fmt(min_val)} to {fmt(max_val)} (mean: {fmt(mean_val)}). The distribution is {skew}."
            
            hist, bin_edges = np.histogram(values, bins=10)
            
            return PowerBIInsight(
                insight_type="distribution",
                measure=measure,
                dimension=None,
                statement=statement,
                chart_type="histogram",
                chart_data={
                    "bin_edges": bin_edges.tolist(),
                    "counts": hist.tolist(),
                    "mean": float(mean_val),
                    "median": float(median_val),
                    "min": float(min_val),
                    "max": float(max_val),
                },
                score=0.65,
            )
        except:
            return None
    
    def _analyze_relationship(
        self,
        df: pl.DataFrame,
        measure1: str,
        measure2: str,
    ) -> Optional[PowerBIInsight]:
        """Answer: Is there a relationship between [measure1] and [measure2]?"""
        try:
            col1 = df[measure1].drop_nulls().to_numpy()
            col2 = df[measure2].drop_nulls().to_numpy()
            
            min_len = min(len(col1), len(col2))
            if min_len < 10:
                return None
            
            col1, col2 = col1[:min_len], col2[:min_len]
            corr = np.corrcoef(col1, col2)[0, 1]
            
            if abs(corr) < 0.3:
                relationship = "weak"
                statement = f"There is no strong relationship between {measure1} and {measure2} (r={corr:.2f})."
            elif corr > 0.7:
                relationship = "strong_positive"
                statement = f"Strong positive relationship: as {measure1} increases, {measure2} increases (r={corr:.2f})."
            elif corr > 0.3:
                relationship = "moderate_positive"
                statement = f"Moderate positive relationship between {measure1} and {measure2} (r={corr:.2f})."
            elif corr < -0.7:
                relationship = "strong_negative"
                statement = f"Strong negative relationship: as {measure1} increases, {measure2} decreases (r={corr:.2f})."
            else:
                relationship = "moderate_negative"
                statement = f"Moderate negative relationship between {measure1} and {measure2} (r={corr:.2f})."
            
            sample_size = min(100, min_len)
            indices = np.random.choice(min_len, sample_size, replace=False)
            
            return PowerBIInsight(
                insight_type="relationship",
                measure=f"{measure1} vs {measure2}",
                dimension=None,
                statement=statement,
                chart_type="scatter",
                chart_data={
                    "x_label": measure1,
                    "y_label": measure2,
                    "x_values": col1[indices].tolist(),
                    "y_values": col2[indices].tolist(),
                    "correlation": round(float(corr), 2),
                    "relationship": relationship,
                },
                score=abs(corr),
            )
        except:
            return None
    
    def _deduplicate_insights(self, insights: list[PowerBIInsight]) -> list[PowerBIInsight]:
        """Keep only the best insight per (measure, insight_type) to ensure diversity."""
        seen = {}  # (measure, insight_type) -> best insight
        
        for insight in insights:
            key = (insight.measure, insight.insight_type)
            
            if key not in seen or insight.score > seen[key].score:
                seen[key] = insight
        
        return list(seen.values())
    
    # ========== 1. Category Outliers (High Value) ==========
    def _find_high_value_insights(
        self,
        df: pl.DataFrame,
        measures: list[str],
        dimensions: list[str],
    ) -> list[PowerBIInsight]:
        """'X' has noticeably more Y."""
        insights = []
        
        for measure in measures[:5]:
            measure_lower = measure.lower()
            rate_keywords = ["price", "cost", "rate", "per", "average", "avg", "margin", "percent", "%"]
            is_rate = any(kw in measure_lower for kw in rate_keywords)
            
            for dim in dimensions[:5]:
                try:
                    if is_rate:
                        agg = (
                            df.group_by(dim)
                            .agg(pl.col(measure).mean().alias("agg_val"))
                            .sort("agg_val", descending=True)
                        )
                    else:
                        agg = (
                            df.group_by(dim)
                            .agg(pl.col(measure).sum().alias("agg_val"))
                            .sort("agg_val", descending=True)
                        )
                    
                    if len(agg) < 2:
                        continue
                    
                    vals = agg["agg_val"].to_list()
                    labels = agg[dim].to_list()
                    
                    top_val = vals[0]
                    second_val = vals[1] if len(vals) > 1 else 0
                    
                    if second_val > 0 and top_val / second_val > 2:
                        agg_word = "higher average" if is_rate else "noticeably more"
                        insights.append(PowerBIInsight(
                            insight_type="high_value",
                            measure=measure,
                            dimension=dim,
                            statement=f"'{labels[0]}' has {agg_word} {measure} than other {dim}s.",
                            chart_type="bar",
                            chart_data={
                                "labels": [str(l) for l in labels[:6]],
                                "values": [float(v) for v in vals[:6]],
                                "highlight": str(labels[0]),
                                "aggregation": "average" if is_rate else "sum",
                            },
                            score=min(1.0, top_val / second_val / 5),
                        ))
                except Exception:
                    continue
        
        return insights
    
    # ========== 2. Change Points in Time Series ==========
    def _find_change_point_insights(
        self,
        df: pl.DataFrame,
        measures: list[str],
        understanding: DataUnderstanding,
    ) -> list[PowerBIInsight]:
        """Significant changes in trends."""
        insights = []
        
        if not understanding.time_columns:
            return insights
        
        time_col = understanding.time_columns[0].name
        
        for measure in measures[:3]:
            try:
                sorted_df = df.sort(time_col)
                values = sorted_df[measure].drop_nulls().to_list()
                
                if len(values) < 20:
                    continue
                
                # Simple change point detection using rolling mean differences
                window = max(5, len(values) // 10)
                changes = []
                
                for i in range(window, len(values) - window):
                    before = np.mean(values[i-window:i])
                    after = np.mean(values[i:i+window])
                    
                    if before != 0:
                        change_pct = abs((after - before) / before) * 100
                        changes.append((i, change_pct, "increase" if after > before else "decrease"))
                
                # Find most significant change point
                if changes:
                    max_change = max(changes, key=lambda x: x[1])
                    if max_change[1] > 20:  # At least 20% change
                        idx = max_change[0]
                        time_labels = sorted_df[time_col].to_list()
                        change_time = str(time_labels[idx]) if idx < len(time_labels) else "mid-series"
                        
                        insights.append(PowerBIInsight(
                            insight_type="change_point",
                            measure=measure,
                            dimension=time_col,
                            statement=f"'{measure}' shows a significant {max_change[2]} around {change_time}.",
                            chart_type="line",
                            chart_data={
                                "values": values[::max(1, len(values)//50)],
                                "change_point_index": idx,
                                "change_percent": round(max_change[1], 1),
                            },
                            score=min(1.0, max_change[1] / 100),
                        ))
            except Exception:
                continue
        
        return insights
    
    # ========== 3. Correlation ==========
    def _find_correlation_insights(
        self,
        df: pl.DataFrame,
        measures: list[str],
    ) -> list[PowerBIInsight]:
        """Strong positive/negative correlation between X and Y."""
        insights = []
        
        if len(measures) < 2:
            return insights
        
        for i, m1 in enumerate(measures[:5]):
            for m2 in measures[i+1:5]:
                try:
                    col1 = df[m1].drop_nulls().to_numpy()
                    col2 = df[m2].drop_nulls().to_numpy()
                    
                    min_len = min(len(col1), len(col2))
                    if min_len < 10:
                        continue
                    
                    col1, col2 = col1[:min_len], col2[:min_len]
                    corr = np.corrcoef(col1, col2)[0, 1]
                    
                    if abs(corr) > 0.5:
                        sample_size = min(100, min_len)
                        indices = np.random.choice(min_len, sample_size, replace=False)
                        
                        # Describe correlation direction and strength
                        if corr > 0.8:
                            strength = "strong positive"
                        elif corr > 0.5:
                            strength = "positive"
                        elif corr < -0.8:
                            strength = "strong negative"
                        else:
                            strength = "negative"
                        
                        insights.append(PowerBIInsight(
                            insight_type="correlation",
                            measure=f"{m1} and {m2}",
                            dimension=None,
                            statement=f"There is a {strength} correlation (r={corr:.2f}) between {m1} and {m2}.",
                            chart_type="scatter",
                            chart_data={
                                "x_label": m1,
                                "y_label": m2,
                                "x_values": col1[indices].tolist(),
                                "y_values": col2[indices].tolist(),
                                "correlation": round(float(corr), 2),
                                "direction": "positive" if corr > 0 else "negative",
                            },
                            score=abs(corr),
                        ))
                except Exception:
                    continue
        
        return insights
    
    # ========== 4. Low Variance ==========
    def _find_low_variance_insights(
        self,
        df: pl.DataFrame,
        measures: list[str],
        dimensions: list[str],
    ) -> list[PowerBIInsight]:
        """There is relatively even Y across all X values."""
        insights = []
        
        for measure in measures[:5]:
            for dim in dimensions[:5]:
                try:
                    agg = (
                        df.group_by(dim)
                        .agg(pl.col(measure).sum().alias("total"))
                        .sort("total", descending=True)
                    )
                    
                    if len(agg) < 3:
                        continue
                    
                    totals = agg["total"].to_list()
                    labels = agg[dim].to_list()
                    
                    mean_val = np.mean(totals)
                    std_val = np.std(totals)
                    
                    if mean_val == 0:
                        continue
                    
                    cv = std_val / mean_val
                    
                    if cv < 0.2:
                        # Calculate min/max for context
                        min_val = min(totals)
                        max_val = max(totals)
                        range_pct = ((max_val - min_val) / mean_val) * 100 if mean_val > 0 else 0
                        
                        insights.append(PowerBIInsight(
                            insight_type="low_variance",
                            measure=measure,
                            dimension=dim,
                            statement=f"{measure} is relatively consistent across all {dim} values (range: {range_pct:.0f}% of mean).",
                            chart_type="bar",
                            chart_data={
                                "labels": [str(l) for l in labels[:10]],
                                "values": [float(v) for v in totals[:10]],
                                "mean": float(mean_val),
                                "range_percent": round(range_pct, 1),
                            },
                            score=1 - cv,
                        ))
                except Exception:
                    continue
        
        return insights
    
    # ========== 5. Majority (Major Factors) ==========
    def _find_majority_insights(
        self,
        df: pl.DataFrame,
        measures: list[str],
        dimensions: list[str],
    ) -> list[PowerBIInsight]:
        """'X' accounts for the majority of Y - only for additive measures."""
        insights = []
        
        # Rate keywords - majority doesn't make sense for these
        rate_keywords = ["price", "cost", "rate", "per", "average", "avg", "margin", "percent", "%"]
        
        for measure in measures[:5]:
            measure_lower = measure.lower()
            
            # Skip rate measures - "majority of Unit Price" doesn't make business sense
            if any(kw in measure_lower for kw in rate_keywords):
                continue
            
            for dim in dimensions[:5]:
                try:
                    agg = (
                        df.group_by(dim)
                        .agg(pl.col(measure).sum().alias("total"))
                        .sort("total", descending=True)
                    )
                    
                    if len(agg) < 2:
                        continue
                    
                    totals = agg["total"].to_list()
                    labels = agg[dim].to_list()
                    grand_total = sum(totals)
                    
                    if grand_total == 0:
                        continue
                    
                    top_pct = (totals[0] / grand_total) * 100
                    
                    if top_pct > 50:
                        insights.append(PowerBIInsight(
                            insight_type="majority",
                            measure=measure,
                            dimension=dim,
                            statement=f"'{labels[0]}' accounts for the majority of {measure} ({top_pct:.0f}%).",
                            chart_type="pie",
                            chart_data={
                                "labels": [str(l) for l in labels[:5]],
                                "values": [float(v) for v in totals[:5]],
                                "percentages": [round(v/grand_total*100, 1) for v in totals[:5]],
                                "highlight": str(labels[0]),
                            },
                            score=top_pct / 100,
                        ))
                except Exception:
                    continue
        
        return insights
    
    # ========== 6. Outliers (Category) ==========
    def _find_outlier_insights(
        self,
        df: pl.DataFrame,
        measures: list[str],
        dimensions: list[str],
    ) -> list[PowerBIInsight]:
        """Y has outliers for X 'A' and 'B'."""
        insights = []
        
        for measure in measures[:5]:
            for dim in dimensions[:3]:
                try:
                    agg = (
                        df.group_by(dim)
                        .agg(pl.col(measure).sum().alias("total"))
                    )
                    
                    if len(agg) < 5:
                        continue
                    
                    totals = agg["total"].to_list()
                    labels = agg[dim].to_list()
                    
                    q1 = np.percentile(totals, 25)
                    q3 = np.percentile(totals, 75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    
                    outlier_labels = [
                        labels[i] for i, v in enumerate(totals)
                        if v < lower or v > upper
                    ]
                    
                    if outlier_labels and len(outlier_labels) <= 3:
                        outlier_str = " and ".join([f"'{o}'" for o in outlier_labels[:2]])
                        
                        insights.append(PowerBIInsight(
                            insight_type="outlier",
                            measure=measure,
                            dimension=dim,
                            statement=f"{measure} has outliers for {dim} {outlier_str}.",
                            chart_type="scatter",
                            chart_data={
                                "labels": [str(l) for l in labels],
                                "values": [float(v) for v in totals],
                                "outliers": [str(o) for o in outlier_labels],
                            },
                            score=0.7,
                        ))
                except Exception:
                    continue
        
        return insights
    
    # ========== 7. Overall Trends ==========
    def _find_trend_insights(
        self,
        df: pl.DataFrame,
        measures: list[str],
        understanding: DataUnderstanding,
    ) -> list[PowerBIInsight]:
        """'Y' is trending upwards/downwards."""
        insights = []
        
        if not understanding.time_columns:
            return insights
        
        time_col = understanding.time_columns[0].name
        
        for measure in measures[:5]:
            try:
                sorted_df = df.sort(time_col)
                values = sorted_df[measure].drop_nulls().to_list()
                
                if len(values) < 10:
                    continue
                
                n = len(values)
                first_third = np.mean(values[:n//3])
                last_third = np.mean(values[-n//3:])
                
                if first_third == 0:
                    continue
                
                change_pct = ((last_third - first_third) / first_third) * 100
                
                if abs(change_pct) > 10:
                    direction = "upwards" if change_pct > 0 else "downwards"
                    sign = "+" if change_pct > 0 else ""
                    
                    insights.append(PowerBIInsight(
                        insight_type="trend",
                        measure=measure,
                        dimension=time_col,
                        statement=f"'{measure}' is trending {direction} ({sign}{change_pct:.0f}% overall change).",
                        chart_type="line",
                        chart_data={
                            "values": values[::max(1, len(values)//50)],
                            "direction": direction,
                            "change_percent": round(change_pct, 1),
                        },
                        score=min(1.0, abs(change_pct) / 50),
                    ))
            except Exception:
                continue
        
        return insights
    
    # ========== 8. Seasonality ==========
    def _find_seasonality_insights(
        self,
        df: pl.DataFrame,
        measures: list[str],
        understanding: DataUnderstanding,
    ) -> list[PowerBIInsight]:
        """Y shows weekly/monthly/yearly seasonality over time."""
        insights = []
        
        if not understanding.time_columns:
            return insights
        
        time_col = understanding.time_columns[0].name
        
        for measure in measures[:3]:
            try:
                # Sort by time and aggregate if needed
                sorted_df = df.sort(time_col)
                
                # Aggregate by time to get time series
                agg_df = (
                    sorted_df.group_by(time_col)
                    .agg(pl.col(measure).sum().alias("value"))
                    .sort(time_col)
                )
                
                values = agg_df["value"].drop_nulls().to_numpy()
                
                if len(values) < 30:
                    continue
                
                # Use FFT to detect periodicity
                fft = np.fft.fft(values - np.mean(values))
                power = np.abs(fft[:len(fft)//2])
                freqs = np.fft.fftfreq(len(values))[:len(values)//2]
                
                # Find dominant frequency (excluding DC component)
                if len(power) < 2:
                    continue
                    
                peak_idx = np.argmax(power[1:]) + 1
                period = int(1 / freqs[peak_idx]) if freqs[peak_idx] > 0 else 0
                
                # Check if periodic signal is strong enough
                if period > 2 and power[peak_idx] > np.mean(power) * 3:
                    # Determine period type
                    if 5 <= period <= 7:
                        period_name = "weekly"
                    elif 28 <= period <= 32:
                        period_name = "monthly"
                    elif 350 <= period <= 370:
                        period_name = "yearly"
                    else:
                        period_name = f"{period}-period"
                    
                    insights.append(PowerBIInsight(
                        insight_type="seasonality",
                        measure=measure,
                        dimension=time_col,
                        statement=f"'{measure}' shows {period_name} seasonality over '{time_col}' (repeating every {period} periods).",
                        chart_type="line",
                        chart_data={
                            "values": values[::max(1, len(values)//50)].tolist(),
                            "period": period,
                            "period_name": period_name,
                            "time_column": time_col,
                        },
                        score=0.65,
                    ))
            except Exception:
                continue
        
        return insights
    
    # ========== 9. Steady Share ==========
    def _find_steady_share_insights(
        self,
        df: pl.DataFrame,
        measures: list[str],
        dimensions: list[str],
        understanding: DataUnderstanding,
    ) -> list[PowerBIInsight]:
        """X has a steady share of Y over time."""
        insights = []
        
        if not understanding.time_columns or not dimensions:
            return insights
        
        time_col = understanding.time_columns[0].name
        
        for measure in measures[:3]:
            for dim in dimensions[:3]:
                try:
                    # Calculate share over time
                    time_dim_agg = (
                        df.group_by([time_col, dim])
                        .agg(pl.col(measure).sum().alias("value"))
                    )
                    
                    time_totals = (
                        df.group_by(time_col)
                        .agg(pl.col(measure).sum().alias("total"))
                    )
                    
                    merged = time_dim_agg.join(time_totals, on=time_col)
                    merged = merged.with_columns(
                        (pl.col("value") / pl.col("total") * 100).alias("share")
                    )
                    
                    # Check each dimension value for steady share
                    for dim_val in df[dim].unique().to_list()[:5]:
                        dim_shares = merged.filter(pl.col(dim) == dim_val)["share"].to_list()
                        
                        if len(dim_shares) < 5:
                            continue
                        
                        mean_share = np.mean(dim_shares)
                        std_share = np.std(dim_shares)
                        
                        if mean_share > 5:  # At least 5% share
                            cv = std_share / mean_share if mean_share > 0 else 1
                            
                            if cv < 0.15:  # Very stable
                                insights.append(PowerBIInsight(
                                    insight_type="steady_share",
                                    measure=measure,
                                    dimension=dim,
                                    statement=f"'{dim_val}' has a steady {mean_share:.0f}% share of {measure} over time.",
                                    chart_type="line",
                                    chart_data={
                                        "dimension_value": str(dim_val),
                                        "share_values": dim_shares,
                                        "mean_share": round(mean_share, 1),
                                    },
                                    score=0.6,
                                ))
                                break  # One per dimension
                except Exception:
                    continue
        
        return insights
    
    # ========== 10. Time Series Outliers ==========
    def _find_time_series_outlier_insights(
        self,
        df: pl.DataFrame,
        measures: list[str],
        understanding: DataUnderstanding,
    ) -> list[PowerBIInsight]:
        """Y has outliers for specific dates."""
        insights = []
        
        if not understanding.time_columns:
            return insights
        
        time_col = understanding.time_columns[0].name
        
        for measure in measures[:3]:
            try:
                sorted_df = df.sort(time_col)
                values = sorted_df[measure].drop_nulls().to_list()
                time_labels = sorted_df[time_col].to_list()
                
                if len(values) < 10:
                    continue
                
                # Find outliers using IQR
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                outlier_times = [
                    str(time_labels[i]) for i, v in enumerate(values)
                    if (v < lower or v > upper) and i < len(time_labels)
                ]
                
                if outlier_times and len(outlier_times) <= 5:
                    outlier_str = " and ".join(outlier_times[:2])
                    
                    insights.append(PowerBIInsight(
                        insight_type="time_series_outlier",
                        measure=measure,
                        dimension=time_col,
                        statement=f"'{measure}' has outliers for {outlier_str}.",
                        chart_type="line",
                        chart_data={
                            "values": values[::max(1, len(values)//50)],
                            "outlier_times": outlier_times,
                        },
                        score=0.7,
                    ))
            except Exception:
                continue
        
        return insights


# Global instance
powerbi_insights_generator = PowerBIInsightsGenerator()

