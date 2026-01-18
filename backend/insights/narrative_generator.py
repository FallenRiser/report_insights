"""
Narrative Generator

Uses LLM to generate executive-friendly narrative insights
from raw statistical findings.
"""

from typing import Any, Optional

import polars as pl

from api.schemas.responses import Insight, InsightType
from llm.ollama_client import ollama_client
from llm.prompts import SYSTEM_PROMPT
from core.logging_config import llm_logger as logger


EXECUTIVE_SUMMARY_PROMPT = """You are a data analyst presenting findings to an executive. Based on the following data analysis results, write a clear, actionable executive summary.

## Dataset Overview
- {row_count:,} records across {column_count} fields
- Dataset type: {dataset_type}
- Primary grouping: {primary_grouper}
- Key measures: {measures}

## Analysis Findings
{findings}

## Your Task
Write a 3-5 paragraph executive summary that:
1. Opens with the single most important finding (the "headline")
2. Explains what patterns were discovered and why they matter
3. Highlights any anomalies or risks that need attention
4. Ends with 2-3 specific, actionable recommendations

Use plain business English. Avoid statistical jargon. Focus on business impact and actions.
Write as if you're speaking to a CEO who has 2 minutes to read this."""


INSIGHT_TO_NARRATIVE_PROMPT = """Convert this statistical insight into a clear business insight:

Type: {insight_type}
Finding: {title}
Details: {description}
Key metrics: {metrics}

Write 1-2 sentences that explain:
1. What this means in business terms
2. Why it matters or what to do about it

Be concise and actionable. No technical jargon."""


class NarrativeGenerator:
    """
    Generates executive-friendly narrative insights using LLM.
    """
    
    async def generate_executive_summary(
        self,
        df: pl.DataFrame,
        insights: list[Insight],
        understanding: Any,
    ) -> str:
        """
        Generate an executive summary from insights.
        
        Args:
            df: The dataset
            insights: List of insights found
            understanding: Data understanding result
            
        Returns:
            Executive summary as narrative text
        """
        logger.info("Generating executive summary with LLM...")
        
        # Format findings for the prompt
        findings_text = self._format_findings(insights)
        logger.debug(f"Formatted {len(insights)} insights for prompt")
        
        # Build prompt
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            row_count=len(df),
            column_count=len(df.columns),
            dataset_type=understanding.dataset_type.value.replace("_", " "),
            primary_grouper=understanding.primary_grouper or "None",
            measures=", ".join(m.name for m in understanding.measure_columns[:5]),
            findings=findings_text,
        )
        
        try:
            logger.debug("Calling LLM for executive summary...")
            summary = await ollama_client.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=1000,
            )
            logger.success(f"LLM summary generated: {len(summary)} chars")
            return summary.strip()
        except Exception as e:
            logger.warning(f"LLM failed, using fallback: {e}")
            # Fallback to simple summary if LLM fails
            return self._generate_fallback_summary(insights, understanding)
    
    async def narrate_insight(self, insight: Insight) -> str:
        """
        Convert a single insight into narrative form.
        """
        prompt = INSIGHT_TO_NARRATIVE_PROMPT.format(
            insight_type=insight.type.value,
            title=insight.title,
            description=insight.description,
            metrics=str(insight.metrics)[:500],
        )
        
        try:
            narrative = await ollama_client.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=200,
            )
            return narrative.strip()
        except Exception:
            return insight.description
    
    def _format_findings(self, insights: list[Insight]) -> str:
        """Format insights as numbered findings."""
        findings = []
        
        for i, insight in enumerate(insights[:10], 1):
            if insight.type == InsightType.SUMMARY:
                continue  # Skip the meta-summary
            
            finding = f"{i}. **{insight.title}**\n   {insight.description}"
            findings.append(finding)
        
        return "\n\n".join(findings) if findings else "No significant findings."
    
    def _generate_fallback_summary(
        self,
        insights: list[Insight],
        understanding: Any,
    ) -> str:
        """Generate summary without LLM (fallback)."""
        
        # Categorize insights
        trends = [i for i in insights if i.type == InsightType.TREND]
        outliers = [i for i in insights if i.type == InsightType.OUTLIER]
        correlations = [i for i in insights if i.type == InsightType.CORRELATION]
        patterns = [i for i in insights if i.type == InsightType.PATTERN]
        
        parts = []
        
        # Headline
        if insights:
            top = next((i for i in insights if i.type != InsightType.SUMMARY), None)
            if top:
                parts.append(f"**Key Finding:** {top.title}\n")
                parts.append(f"{top.description}\n")
        
        # Correlations
        if correlations:
            c = correlations[0]
            parts.append(f"\n**Relationship Discovered:** {c.columns[0]} and {c.columns[1]} "
                        f"move together (correlation: {c.metrics.get('pearson', 0):.0%}). "
                        f"Changes to one will likely affect the other.\n")
        
        # Patterns
        if patterns:
            parts.append(f"\n**Data Patterns:** Found {len(patterns)} notable patterns. ")
            for p in patterns[:2]:
                parts.append(f"{p.title}. ")
            parts.append("\n")
        
        # Outliers
        if outliers:
            parts.append(f"\n**Attention Needed:** {len(outliers)} outlier situations detected "
                        f"that may require investigation.\n")
        
        # Recommendations based on understanding
        if understanding.primary_grouper:
            parts.append(f"\n**Recommendation:** Analyze performance by '{understanding.primary_grouper}' "
                        f"to identify best and worst performing segments.\n")
        
        return "".join(parts)
    
    def generate_quick_narrative(
        self,
        insights: list[Insight],
        understanding: Any,
    ) -> str:
        """
        Generate a quick narrative summary without LLM.
        
        This is fast and doesn't require Ollama to be running.
        """
        lines = []
        
        # Opening - what kind of data
        dtype = understanding.dataset_type.value.replace("_", " ")
        lines.append(f"## Executive Summary\n")
        lines.append(f"Analysis of your {dtype} data with {understanding.row_count:,} records reveals:\n")
        
        # Top findings - skip summary type
        top_findings = [i for i in insights if i.type != InsightType.SUMMARY][:5]
        
        if not top_findings:
            lines.append("\n**No significant patterns detected.** The data appears stable without notable anomalies.")
            return "\n".join(lines)
        
        lines.append("\n### Key Findings\n")
        
        for i, insight in enumerate(top_findings, 1):
            # Convert type to business language
            type_labels = {
                InsightType.TREND: "📈 Trend",
                InsightType.OUTLIER: "⚠️ Anomaly",
                InsightType.CORRELATION: "🔗 Relationship",
                InsightType.PATTERN: "📊 Pattern",
                InsightType.DISTRIBUTION: "📊 Distribution",
                InsightType.SEASONALITY: "🔄 Seasonal Pattern",
            }
            label = type_labels.get(insight.type, "Finding")
            
            # Create business-friendly description
            lines.append(f"**{i}. {label}:** {self._businessify(insight)}\n")
        
        # Recommendations
        lines.append("\n### Recommended Actions\n")
        
        recommendations = self._generate_recommendations(top_findings, understanding)
        for rec in recommendations[:3]:
            lines.append(f"- {rec}\n")
        
        return "\n".join(lines)
    
    def _businessify(self, insight: Insight) -> str:
        """Convert insight to business language."""
        
        if insight.type == InsightType.CORRELATION:
            col1 = insight.columns[0] if insight.columns else "Variable 1"
            col2 = insight.columns[1] if len(insight.columns) > 1 else "Variable 2"
            corr = insight.metrics.get("pearson", 0)
            
            if corr > 0:
                return f"{col1} and {col2} increase together. When one goes up, the other typically follows. This {abs(corr):.0%} correlation suggests they're closely linked."
            else:
                return f"{col1} and {col2} move in opposite directions. When one increases, the other tends to decrease."
        
        elif insight.type == InsightType.OUTLIER:
            col = insight.columns[0] if insight.columns else "the data"
            count = insight.metrics.get("outlier_count", 0)
            return f"{count} unusual values found in {col} that fall outside normal ranges. These may be errors or exceptional cases worth investigating."
        
        elif insight.type == InsightType.PATTERN:
            return insight.description
        
        elif insight.type == InsightType.DISTRIBUTION:
            col = insight.columns[0] if insight.columns else "the data"
            modality = insight.metrics.get("modality", 1)
            if modality >= 2:
                return f"{col} shows {modality} distinct groupings in the data, suggesting different subpopulations or categories may be mixed together."
            return insight.description
        
        return insight.description
    
    def _generate_recommendations(
        self,
        insights: list[Insight],
        understanding: Any,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs = []
        
        # Based on grouping
        if understanding.primary_grouper:
            recs.append(
                f"Segment your analysis by '{understanding.primary_grouper}' to compare performance across groups"
            )
        
        # Based on insight types
        has_correlation = any(i.type == InsightType.CORRELATION for i in insights)
        has_outlier = any(i.type == InsightType.OUTLIER for i in insights)
        has_distribution = any(i.type == InsightType.DISTRIBUTION for i in insights)
        
        if has_correlation:
            corr_insight = next(i for i in insights if i.type == InsightType.CORRELATION)
            recs.append(
                f"Investigate the relationship between {corr_insight.columns[0]} and {corr_insight.columns[1]} - "
                f"changes to one may impact the other"
            )
        
        if has_outlier:
            recs.append(
                "Review the flagged outliers to determine if they're data errors or genuine exceptional cases"
            )
        
        if has_distribution:
            dist_insight = next(i for i in insights if i.type == InsightType.DISTRIBUTION)
            if dist_insight.metrics.get("modality", 1) >= 2:
                recs.append(
                    f"Consider segmenting {dist_insight.columns[0]} into distinct groups for more targeted analysis"
                )
        
        # Default recommendation
        if not recs:
            recs.append("Continue monitoring key metrics for emerging patterns")
        
        return recs[:4]


# Global instance
narrative_generator = NarrativeGenerator()
