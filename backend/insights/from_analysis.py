"""
Insight Generators from Analysis

Generates insights FROM analysis results, ensuring they are
backed by actual statistical analysis.
"""

from typing import Optional
from dataclasses import dataclass, field
from analysis.orchestrator import AnalysisResults
from core.data_understanding import DataUnderstanding
from core.logging_config import insights_logger as logger


@dataclass
class AnalysisDrivenInsight:
    """
    Insight derived from statistical analysis.
    """
    insight_type: str
    measure: str
    dimension: Optional[str]
    statement: str
    chart_type: str
    chart_data: dict
    score: float
    # What analysis backs this insight
    analysis_source: str
    supporting_data: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "insight_type": self.insight_type,
            "measure": self.measure,
            "dimension": self.dimension,
            "statement": self.statement,
            "chart_type": self.chart_type,
            "chart_data": self.chart_data,
            "score": self.score,
            "analysis_source": self.analysis_source,
        }


class InsightGeneratorFromAnalysis:
    """
    Generates insights FROM analysis results.
    
    This ensures every insight is backed by actual statistical analysis,
    not just pattern matching on raw data.
    """
    
    def __init__(self):
        self.logger = logger
    
    def generate_all(
        self,
        analysis: AnalysisResults,
        understanding: DataUnderstanding,
        max_insights: int = 30,
    ) -> list[AnalysisDrivenInsight]:
        """
        Generate insights from all analysis results.
        """
        self.logger.info("=== GENERATING INSIGHTS FROM ANALYSIS ===")
        
        all_insights = []
        
        # 1. Insights from KEY INFLUENCERS
        if analysis.key_influencers:
            self.logger.info("Generating key driver insights...")
            ki_insights = self._from_key_influencers(analysis.key_influencers, understanding)
            all_insights.extend(ki_insights)
            self.logger.success(f"Generated {len(ki_insights)} key driver insights")
        
        # 2. Insights from CORRELATIONS
        if analysis.correlations:
            self.logger.info("Generating correlation insights...")
            corr_insights = self._from_correlations(analysis.correlations, understanding)
            all_insights.extend(corr_insights)
            self.logger.success(f"Generated {len(corr_insights)} correlation insights")
        
        # 3. Insights from DECOMPOSITION
        if analysis.decomposition:
            self.logger.info("Generating breakdown insights...")
            decomp_insights = self._from_decomposition(analysis.decomposition, understanding)
            all_insights.extend(decomp_insights)
            self.logger.success(f"Generated {len(decomp_insights)} breakdown insights")
        
        # 4. Insights from OUTLIERS
        if analysis.outliers:
            self.logger.info("Generating outlier insights...")
            outlier_insights = self._from_outliers(analysis.outliers, understanding)
            all_insights.extend(outlier_insights)
            self.logger.success(f"Generated {len(outlier_insights)} outlier insights")
        
        # 5. Insights from TRENDS
        if analysis.trends:
            self.logger.info("Generating trend insights...")
            trend_insights = self._from_trends(analysis.trends, understanding)
            all_insights.extend(trend_insights)
            self.logger.success(f"Generated {len(trend_insights)} trend insights")
        
        # 6. Insights from DISTRIBUTIONS
        if analysis.distributions:
            self.logger.info("Generating distribution insights...")
            dist_insights = self._from_distributions(analysis.distributions, understanding)
            all_insights.extend(dist_insights)
            self.logger.success(f"Generated {len(dist_insights)} distribution insights")
        
        # Sort by score and return top N
        all_insights.sort(key=lambda x: x.score, reverse=True)
        final_insights = all_insights[:max_insights]
        
        self.logger.success(f"=== GENERATED {len(final_insights)} INSIGHTS FROM ANALYSIS ===")
        return final_insights
    
    def _from_key_influencers(
        self, 
        ki: dict, 
        understanding: DataUnderstanding
    ) -> list[AnalysisDrivenInsight]:
        """Generate insights from key influencers analysis."""
        insights = []
        
        target = ki.get("target_column", "the measure")
        influencers = ki.get("influencers", [])
        
        for i, inf in enumerate(influencers[:5]):
            feature = inf.get("feature", "Unknown")
            importance = inf.get("importance", 0)
            direction = inf.get("direction", "positive")
            
            # Skip if importance is too low
            if importance < 0.05:
                continue
            
            # Create contextual statement
            if direction == "positive":
                effect = "increases"
            elif direction == "negative":
                effect = "decreases"
            else:
                effect = "affects"
            
            statement = f"'{feature}' is a key driver of {target} ({importance:.0%} importance). Higher values {effect} {target}."
            
            insights.append(AnalysisDrivenInsight(
                insight_type="key_driver",
                measure=target,
                dimension=feature,
                statement=statement,
                chart_type="bar",
                chart_data={
                    "feature": feature,
                    "importance": importance,
                    "direction": direction,
                    "rank": i + 1,
                },
                score=0.9 - (i * 0.1),  # Top influencer gets highest score
                analysis_source="key_influencers",
                supporting_data={"shap_mean": inf.get("shap_mean", 0)},
            ))
        
        return insights
    
    def _from_correlations(
        self, 
        correlations: list[dict], 
        understanding: DataUnderstanding
    ) -> list[AnalysisDrivenInsight]:
        """Generate insights from correlation analysis."""
        insights = []
        
        for corr in correlations[:10]:
            col1 = corr.get("column1", "X")
            col2 = corr.get("column2", "Y")
            r = corr.get("pearson", corr.get("correlation", 0))
            direction = corr.get("direction", "positive" if r > 0 else "negative")
            
            # Only strong correlations become insights
            if abs(r) < 0.5:
                continue
            
            strength = "strongly" if abs(r) > 0.7 else "moderately"
            
            if direction == "positive":
                relationship = "increase together"
            else:
                relationship = "move in opposite directions"
            
            statement = f"'{col1}' and '{col2}' are {strength} correlated (r={r:.2f}). They {relationship}."
            
            insights.append(AnalysisDrivenInsight(
                insight_type="correlation",
                measure=col1,
                dimension=col2,
                statement=statement,
                chart_type="scatter",
                chart_data={
                    "x_column": col1,
                    "y_column": col2,
                    "correlation": r,
                    "direction": direction,
                },
                score=abs(r),
                analysis_source="correlations",
            ))
        
        return insights
    
    def _from_decomposition(
        self, 
        decomp: dict, 
        understanding: DataUnderstanding
    ) -> list[AnalysisDrivenInsight]:
        """Generate insights from decomposition analysis."""
        insights = []
        
        measure = decomp.get("measure_column", "the measure")
        breakdown = decomp.get("breakdown", {})
        
        for dim_name, dim_data in breakdown.items():
            if not isinstance(dim_data, dict):
                continue
            
            contributions = dim_data.get("contributions", [])
            if not contributions:
                continue
            
            # Find the top contributor
            top = contributions[0] if contributions else None
            if not top:
                continue
            
            value = top.get("value", "Unknown")
            percentage = top.get("percentage", 0)
            
            # Only create insight if contribution is significant
            if percentage < 20:
                continue
            
            statement = f"'{value}' accounts for {percentage:.0f}% of {measure} across {dim_name}."
            
            insights.append(AnalysisDrivenInsight(
                insight_type="majority",
                measure=measure,
                dimension=dim_name,
                statement=statement,
                chart_type="pie",
                chart_data={
                    "dimension": dim_name,
                    "top_value": value,
                    "percentage": percentage,
                    "all_contributions": contributions[:5],
                },
                score=percentage / 100,
                analysis_source="decomposition",
            ))
        
        return insights
    
    def _from_outliers(
        self, 
        outliers: dict[str, dict], 
        understanding: DataUnderstanding
    ) -> list[AnalysisDrivenInsight]:
        """Generate insights from outlier analysis."""
        insights = []
        
        for measure, outlier_data in outliers.items():
            count = outlier_data.get("outlier_count", 0)
            percentage = outlier_data.get("outlier_percentage", 0)
            
            if count == 0:
                continue
            
            severity = "extreme" if percentage > 5 else "notable"
            
            statement = f"{count} {severity} outliers detected in '{measure}' ({percentage:.1f}% of data)."
            
            insights.append(AnalysisDrivenInsight(
                insight_type="outlier",
                measure=measure,
                dimension=None,
                statement=statement,
                chart_type="box",
                chart_data={
                    "measure": measure,
                    "outlier_count": count,
                    "outlier_percentage": percentage,
                    "bounds": {
                        "lower": outlier_data.get("lower_bound"),
                        "upper": outlier_data.get("upper_bound"),
                    }
                },
                score=min(1.0, percentage / 10),
                analysis_source="outliers",
            ))
        
        return insights
    
    def _from_trends(
        self, 
        trends: dict[str, dict], 
        understanding: DataUnderstanding
    ) -> list[AnalysisDrivenInsight]:
        """Generate insights from trend analysis."""
        insights = []
        
        for measure, trend_data in trends.items():
            direction = trend_data.get("direction", "stable")
            change_pct = trend_data.get("change_percent", 0)
            
            if direction == "stable" or abs(change_pct) < 5:
                continue
            
            trend_word = "up" if direction == "increasing" else "down"
            
            statement = f"'{measure}' is trending {trend_word} ({change_pct:+.1f}% change)."
            
            insights.append(AnalysisDrivenInsight(
                insight_type="trend",
                measure=measure,
                dimension=None,
                statement=statement,
                chart_type="line",
                chart_data={
                    "measure": measure,
                    "direction": direction,
                    "change_percent": change_pct,
                    "slope": trend_data.get("slope", 0),
                },
                score=min(1.0, abs(change_pct) / 50),
                analysis_source="trends",
            ))
        
        return insights
    
    def _from_distributions(
        self, 
        distributions: dict[str, dict], 
        understanding: DataUnderstanding
    ) -> list[AnalysisDrivenInsight]:
        """Generate insights from distribution analysis."""
        insights = []
        
        for measure, dist_data in distributions.items():
            skewness = dist_data.get("skewness", 0)
            modality = dist_data.get("modality", 1)
            
            # Only create insight for notable distributions
            if abs(skewness) < 0.5 and modality <= 1:
                continue
            
            if abs(skewness) > 1:
                skew_desc = "right-skewed" if skewness > 0 else "left-skewed"
                statement = f"'{measure}' is highly {skew_desc} (skewness={skewness:.2f})."
                chart_type = "histogram"
            elif modality >= 2:
                statement = f"'{measure}' has {modality} distinct peaks, suggesting multiple subgroups in the data."
                chart_type = "histogram"
            else:
                continue
            
            insights.append(AnalysisDrivenInsight(
                insight_type="distribution",
                measure=measure,
                dimension=None,
                statement=statement,
                chart_type=chart_type,
                chart_data={
                    "measure": measure,
                    "skewness": skewness,
                    "modality": modality,
                    "mean": dist_data.get("mean"),
                    "median": dist_data.get("median"),
                },
                score=min(1.0, (abs(skewness) + modality) / 5),
                analysis_source="distributions",
            ))
        
        return insights


# Global instance
insight_generator_from_analysis = InsightGeneratorFromAnalysis()
