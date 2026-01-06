"""
Insight Ranker

Ranks and prioritizes insights based on statistical significance,
novelty, and business relevance.
"""

from dataclasses import dataclass
from typing import Any

from api.schemas.responses import Insight, InsightSeverity, InsightType


@dataclass
class RankingCriteria:
    """Criteria weights for ranking insights."""
    
    significance_weight: float = 0.35
    novelty_weight: float = 0.25
    magnitude_weight: float = 0.20
    coverage_weight: float = 0.10
    actionability_weight: float = 0.10


class InsightRanker:
    """
    Ranks insights by importance and relevance.
    
    Uses multiple criteria to surface the most valuable insights first.
    """
    
    def __init__(self, criteria: RankingCriteria = None):
        self.criteria = criteria or RankingCriteria()
    
    def rank_insights(
        self,
        insights: list[Insight],
        top_n: int = 10,
    ) -> list[Insight]:
        """
        Rank and return top N insights.
        
        Args:
            insights: List of insights to rank
            top_n: Number of top insights to return
            
        Returns:
            Sorted list of top insights
        """
        if not insights:
            return []
        
        # Calculate composite scores
        scored_insights = []
        for insight in insights:
            score = self._calculate_score(insight)
            scored_insights.append((insight, score))
        
        # Sort by score descending
        scored_insights.sort(key=lambda x: x[1], reverse=True)
        
        # Update insight scores and return top N
        result = []
        for insight, score in scored_insights[:top_n]:
            insight.score = score
            result.append(insight)
        
        return result
    
    def _calculate_score(self, insight: Insight) -> float:
        """Calculate composite ranking score for an insight."""
        
        # Extract score components from metrics
        metrics = insight.metrics or {}
        
        # Significance score (0-1)
        # Based on p-value if available, or direct score
        significance = self._score_significance(insight, metrics)
        
        # Novelty score (0-1)
        # Unexpected or unusual findings score higher
        novelty = self._score_novelty(insight, metrics)
        
        # Magnitude score (0-1)
        # Large changes or deviations score higher
        magnitude = self._score_magnitude(insight, metrics)
        
        # Coverage score (0-1)
        # Insights affecting more data score higher
        coverage = self._score_coverage(insight, metrics)
        
        # Actionability score (0-1)
        # Insights that suggest clear actions score higher
        actionability = self._score_actionability(insight)
        
        # Weighted sum
        score = (
            self.criteria.significance_weight * significance +
            self.criteria.novelty_weight * novelty +
            self.criteria.magnitude_weight * magnitude +
            self.criteria.coverage_weight * coverage +
            self.criteria.actionability_weight * actionability
        )
        
        return min(1.0, max(0.0, score))
    
    def _score_significance(
        self,
        insight: Insight,
        metrics: dict[str, Any],
    ) -> float:
        """Score based on statistical significance."""
        
        # Check for p-value
        p_value = metrics.get("p_value")
        if p_value is not None:
            # Lower p-value = higher significance
            if p_value < 0.001:
                return 1.0
            elif p_value < 0.01:
                return 0.9
            elif p_value < 0.05:
                return 0.7
            elif p_value < 0.1:
                return 0.5
            else:
                return 0.2
        
        # Check for r-squared
        r_squared = metrics.get("r_squared")
        if r_squared is not None:
            return min(1.0, r_squared)
        
        # Use severity as fallback
        severity_scores = {
            InsightSeverity.CRITICAL: 1.0,
            InsightSeverity.HIGH: 0.8,
            InsightSeverity.MEDIUM: 0.5,
            InsightSeverity.LOW: 0.3,
        }
        return severity_scores.get(insight.severity, 0.5)
    
    def _score_novelty(
        self,
        insight: Insight,
        metrics: dict[str, Any],
    ) -> float:
        """Score based on unexpectedness."""
        
        # Outliers and anomalies are novel
        if insight.type in (InsightType.OUTLIER, InsightType.ANOMALY):
            outlier_pct = metrics.get("outlier_percentage", 5)
            # Rare outliers are more novel
            if outlier_pct < 1:
                return 1.0
            elif outlier_pct < 5:
                return 0.8
            else:
                return 0.5
        
        # Unexpected correlations are novel
        if insight.type == InsightType.CORRELATION:
            # Strong correlations are less expected
            corr = abs(metrics.get("correlation", 0))
            if corr > 0.8:
                return 0.9
            elif corr > 0.6:
                return 0.7
            else:
                return 0.4
        
        # Trend changes are novel
        if insight.type == InsightType.TREND:
            if "change_points" in metrics and metrics["change_points"]:
                return 0.8
            return 0.5
        
        return 0.5
    
    def _score_magnitude(
        self,
        insight: Insight,
        metrics: dict[str, Any],
    ) -> float:
        """Score based on size of effect."""
        
        # Percent change
        pct_change = metrics.get("percent_change")
        if pct_change is not None:
            abs_pct = abs(pct_change)
            if abs_pct > 100:
                return 1.0
            elif abs_pct > 50:
                return 0.8
            elif abs_pct > 20:
                return 0.6
            elif abs_pct > 10:
                return 0.4
            else:
                return 0.2
        
        # Correlation strength
        if insight.type == InsightType.CORRELATION:
            return abs(metrics.get("correlation", 0))
        
        # Feature importance
        importance = metrics.get("importance")
        if importance is not None:
            return min(1.0, importance)
        
        return 0.5
    
    def _score_coverage(
        self,
        insight: Insight,
        metrics: dict[str, Any],
    ) -> float:
        """Score based on how much data is affected."""
        
        # Data points affected
        count = metrics.get("count", 0)
        total = metrics.get("total_count", count)
        
        if total > 0 and count > 0:
            coverage = count / total
            return min(1.0, coverage)
        
        # Columns involved
        n_columns = len(insight.columns)
        if n_columns >= 5:
            return 0.8
        elif n_columns >= 3:
            return 0.6
        elif n_columns >= 2:
            return 0.4
        
        return 0.3
    
    def _score_actionability(
        self,
        insight: Insight,
    ) -> float:
        """Score based on whether the insight suggests actions."""
        
        # Type-based actionability
        actionable_types = {
            InsightType.OUTLIER: 0.9,  # Can investigate/fix
            InsightType.ANOMALY: 0.9,
            InsightType.TREND: 0.7,  # Can project/plan
            InsightType.CORRELATION: 0.6,  # Can investigate causation
            InsightType.KEY_INFLUENCER: 0.8,  # Can focus efforts
            InsightType.PATTERN: 0.5,
            InsightType.DISTRIBUTION: 0.4,
            InsightType.SEASONALITY: 0.7,  # Can plan around
            InsightType.SUMMARY: 0.3,
        }
        
        return actionable_types.get(insight.type, 0.5)
    
    def diversify_insights(
        self,
        insights: list[Insight],
        max_per_type: int = 3,
        max_per_column: int = 2,
    ) -> list[Insight]:
        """
        Diversify insights to avoid redundancy.
        
        Limits insights per type and per column.
        """
        result = []
        type_counts: dict[InsightType, int] = {}
        column_counts: dict[str, int] = {}
        
        for insight in insights:
            # Check type limit
            type_count = type_counts.get(insight.type, 0)
            if type_count >= max_per_type:
                continue
            
            # Check column limit
            skip = False
            for col in insight.columns:
                if column_counts.get(col, 0) >= max_per_column:
                    skip = True
                    break
            
            if skip:
                continue
            
            # Add insight
            result.append(insight)
            type_counts[insight.type] = type_count + 1
            for col in insight.columns:
                column_counts[col] = column_counts.get(col, 0) + 1
        
        return result
    
    def determine_severity(
        self,
        p_value: float = None,
        magnitude: float = None,
        outlier_percentage: float = None,
    ) -> InsightSeverity:
        """
        Determine insight severity from metrics.
        """
        scores = []
        
        if p_value is not None:
            if p_value < 0.001:
                scores.append(4)
            elif p_value < 0.01:
                scores.append(3)
            elif p_value < 0.05:
                scores.append(2)
            else:
                scores.append(1)
        
        if magnitude is not None:
            if magnitude > 0.8:
                scores.append(4)
            elif magnitude > 0.5:
                scores.append(3)
            elif magnitude > 0.3:
                scores.append(2)
            else:
                scores.append(1)
        
        if outlier_percentage is not None:
            if outlier_percentage < 1:
                scores.append(4)
            elif outlier_percentage < 3:
                scores.append(3)
            elif outlier_percentage < 10:
                scores.append(2)
            else:
                scores.append(1)
        
        if not scores:
            return InsightSeverity.MEDIUM
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 3.5:
            return InsightSeverity.CRITICAL
        elif avg_score >= 2.5:
            return InsightSeverity.HIGH
        elif avg_score >= 1.5:
            return InsightSeverity.MEDIUM
        else:
            return InsightSeverity.LOW


# Global instance
insight_ranker = InsightRanker()
