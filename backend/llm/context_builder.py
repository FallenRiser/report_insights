"""
Context Builder

Builds data context for LLM prompts with token budget management.
"""

from typing import Any, Optional

import polars as pl

from core.data_profiler import data_profiler
from api.schemas.responses import DataProfile, Insight


class ContextBuilder:
    """
    Builds context strings for LLM prompts.
    
    Manages token budget to avoid exceeding limits while
    providing maximum relevant information.
    """
    
    MAX_CONTEXT_CHARS = 8000  # Approximately 2000 tokens
    
    def __init__(self, max_chars: int = None):
        self.max_chars = max_chars or self.MAX_CONTEXT_CHARS
    
    def build_data_context(
        self,
        df: pl.DataFrame,
        profile: Optional[DataProfile] = None,
    ) -> str:
        """
        Build data context string for prompts.
        
        Includes schema, basic stats, and sample data.
        """
        if profile is None:
            profile = data_profiler.profile(df)
        
        parts = []
        
        # Basic info
        parts.append(f"Dataset: {profile.row_count:,} rows × {profile.column_count} columns")
        parts.append(f"Memory: {profile.memory_usage_mb:.1f} MB")
        parts.append("")
        
        # Column summary
        parts.append("## Columns")
        for col_profile in profile.columns[:20]:  # Limit columns
            dtype = col_profile.dtype.replace("polars.", "")
            null_pct = f"({col_profile.null_percentage:.0f}% null)" if col_profile.null_percentage > 0 else ""
            
            if col_profile.mean is not None:
                stats = f"mean={col_profile.mean:.2f}, std={col_profile.std:.2f}"
            elif col_profile.top_values:
                top = col_profile.top_values[0]
                stats = f"top='{top['value']}' ({top['percentage']:.0f}%)"
            else:
                stats = ""
            
            parts.append(f"- **{col_profile.name}** ({dtype}) {null_pct}: {stats}")
        
        if len(profile.columns) > 20:
            parts.append(f"  ... and {len(profile.columns) - 20} more columns")
        
        # Sample data
        parts.append("")
        parts.append("## Sample Data (first 5 rows)")
        sample = df.head(5)
        
        # Format as simple table
        headers = sample.columns[:10]
        parts.append("| " + " | ".join(headers) + " |")
        parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in sample.iter_rows():
            values = [str(v)[:20] if v is not None else "null" for v in row[:10]]
            parts.append("| " + " | ".join(values) + " |")
        
        context = "\n".join(parts)
        return self._truncate(context)
    
    def build_insights_context(
        self,
        insights: list[Insight],
        max_insights: int = 5,
    ) -> str:
        """Build context from previous insights."""
        if not insights:
            return "No insights generated yet."
        
        parts = ["## Key Insights"]
        
        for insight in insights[:max_insights]:
            parts.append(f"- **{insight.title}**: {insight.description[:200]}")
        
        if len(insights) > max_insights:
            parts.append(f"  ... and {len(insights) - max_insights} more insights")
        
        return "\n".join(parts)
    
    def build_column_context(
        self,
        df: pl.DataFrame,
        column: str,
    ) -> str:
        """Build detailed context for a specific column."""
        col = df[column]
        
        parts = [f"## Column: {column}"]
        parts.append(f"Type: {col.dtype}")
        parts.append(f"Count: {len(col):,} ({col.null_count()} nulls)")
        parts.append(f"Unique values: {col.n_unique()}")
        
        if col.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            non_null = col.drop_nulls()
            parts.append(f"Range: {non_null.min()} to {non_null.max()}")
            parts.append(f"Mean: {non_null.mean():.4f}")
            parts.append(f"Median: {non_null.median():.4f}")
            parts.append(f"Std: {non_null.std():.4f}")
        else:
            # Categorical - show top values
            value_counts = col.value_counts().sort("count", descending=True).head(10)
            parts.append("Top values:")
            for row in value_counts.iter_rows(named=True):
                value_col = [c for c in value_counts.columns if c != "count"][0]
                parts.append(f"  - '{row[value_col]}': {row['count']:,}")
        
        return "\n".join(parts)
    
    def build_comparison_context(
        self,
        df: pl.DataFrame,
        column1: str,
        column2: str,
    ) -> str:
        """Build context for comparing two columns."""
        parts = [f"## Comparison: {column1} vs {column2}"]
        
        col1 = df[column1].drop_nulls()
        col2 = df[column2].drop_nulls()
        
        parts.append(f"### {column1}")
        parts.append(f"Type: {col1.dtype}, Count: {len(col1):,}")
        if col1.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            parts.append(f"Mean: {col1.mean():.4f}, Std: {col1.std():.4f}")
        
        parts.append(f"### {column2}")
        parts.append(f"Type: {col2.dtype}, Count: {len(col2):,}")
        if col2.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            parts.append(f"Mean: {col2.mean():.4f}, Std: {col2.std():.4f}")
        
        # Correlation if both numeric
        if (col1.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32) and
            col2.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)):
            from analysis.correlations import correlation_analyzer
            corr = correlation_analyzer.compute_correlation_pair(df, column1, column2)
            parts.append(f"### Correlation")
            parts.append(f"Pearson: {corr.pearson:.4f}")
            parts.append(f"Spearman: {corr.spearman:.4f}")
        
        return "\n".join(parts)
    
    def build_query_context(
        self,
        df: pl.DataFrame,
        profile: DataProfile,
        insights: list[Insight],
        query: str,
    ) -> dict[str, str]:
        """Build full context for a conversational query."""
        return {
            "data_context": self.build_data_context(df, profile),
            "insights_context": self.build_insights_context(insights),
            "user_query": query,
        }
    
    def _truncate(self, text: str) -> str:
        """Truncate text to max chars."""
        if len(text) <= self.max_chars:
            return text
        
        return text[:self.max_chars - 50] + "\n\n... (truncated for length)"


# Global instance
context_builder = ContextBuilder()
