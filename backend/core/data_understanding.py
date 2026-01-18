"""
Data Understanding Layer

Intelligently analyzes data structure and semantics before running analysis.
Determines:
- What type of data this is (time series, transactional, hierarchical, etc.)
- What grouping columns exist (routes, categories, regions)
- What the appropriate analysis strategies are
- Whether data should be analyzed globally or by groups
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

import numpy as np
import polars as pl

from core.data_profiler import data_profiler


class DatasetType(str, Enum):
    """Types of datasets detected."""
    
    TIME_SERIES = "time_series"           # Sequential data over time
    TRANSACTIONAL = "transactional"       # Individual events/transactions
    HIERARCHICAL = "hierarchical"         # Grouped/nested structure
    CROSS_SECTIONAL = "cross_sectional"   # Snapshot at a point in time
    PANEL = "panel"                       # Both time series and cross-sectional
    UNKNOWN = "unknown"


class AnalysisScope(str, Enum):
    """How to scope the analysis."""
    
    GLOBAL = "global"       # Analyze entire dataset
    BY_GROUP = "by_group"   # Analyze within each group
    BOTH = "both"           # Do both global and grouped analysis


@dataclass
class GroupingColumn:
    """A column identified as useful for grouping."""
    
    name: str
    unique_values: int
    value_distribution: dict[str, int]  # Top values
    is_primary: bool  # Whether this seems like a primary grouper
    coverage: float   # Percentage of data covered by top groups
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unique_values": self.unique_values,
            "value_distribution": self.value_distribution,
            "is_primary": self.is_primary,
            "coverage": round(self.coverage, 2),
        }


@dataclass
class TimeColumn:
    """A column identified as temporal."""
    
    name: str
    is_sorted: bool
    has_gaps: bool
    frequency: Optional[str]  # daily, hourly, monthly, etc.
    date_range: Optional[tuple[str, str]]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_sorted": self.is_sorted,
            "has_gaps": self.has_gaps,
            "frequency": self.frequency,
            "date_range": self.date_range,
        }


@dataclass
class MeasureColumn:
    """A column identified as a measure to analyze."""
    
    name: str
    analysis_type: str  # numeric, count, ratio, currency, etc.
    aggregatable: bool  # Can be summed/averaged meaningfully
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "analysis_type": self.analysis_type,
            "aggregatable": self.aggregatable,
        }


@dataclass
class StoryColumn:
    """A column identified as important for storytelling."""
    
    name: str
    role: str  # "primary_measure", "secondary_measure", "key_dimension", "time_axis"
    business_meaning: str  # What this column represents in business terms
    importance: float  # 0-1 score
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "business_meaning": self.business_meaning,
            "importance": round(self.importance, 2),
        }


@dataclass
class AnalysisRecommendation:
    """Recommendation for how to analyze the data."""
    
    analysis_name: str
    applicable: bool
    reason: str
    scope: AnalysisScope
    target_columns: list[str]
    group_by: Optional[str] = None
    priority: int = 5  # 1-10, higher = more important
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_name": self.analysis_name,
            "applicable": self.applicable,
            "reason": self.reason,
            "scope": self.scope.value,
            "target_columns": self.target_columns,
            "group_by": self.group_by,
            "priority": self.priority,
        }


@dataclass
class DataUnderstanding:
    """Complete understanding of the dataset."""
    
    dataset_type: DatasetType
    row_count: int
    column_count: int
    
    # Identified columns
    time_columns: list[TimeColumn]
    grouping_columns: list[GroupingColumn]
    measure_columns: list[MeasureColumn]
    id_columns: list[str]  # Identifiers (not useful for analysis)
    
    # Analysis strategy
    recommended_scope: AnalysisScope
    primary_grouper: Optional[str]
    analysis_recommendations: list[AnalysisRecommendation]
    
    # Data quality
    missing_data_severity: str  # low, medium, high
    data_quality_issues: list[str]
    
    # SEMANTIC UNDERSTANDING - What is this data about?
    business_domain: str = "general"  # e.g., "sales", "finance", "hr", "marketing"
    data_story: str = ""  # Brief description of what this data represents
    key_questions: list[str] = field(default_factory=list)  # Business questions this data can answer
    story_columns: list[StoryColumn] = field(default_factory=list)  # Most important columns for storytelling
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_type": self.dataset_type.value,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "time_columns": [c.to_dict() for c in self.time_columns],
            "grouping_columns": [c.to_dict() for c in self.grouping_columns],
            "measure_columns": [c.to_dict() for c in self.measure_columns],
            "id_columns": self.id_columns,
            "recommended_scope": self.recommended_scope.value,
            "primary_grouper": self.primary_grouper,
            "analysis_recommendations": [r.to_dict() for r in self.analysis_recommendations],
            "missing_data_severity": self.missing_data_severity,
            "data_quality_issues": self.data_quality_issues,
            # Semantic understanding
            "business_domain": self.business_domain,
            "data_story": self.data_story,
            "key_questions": self.key_questions,
            "story_columns": [c.to_dict() for c in self.story_columns],
        }


class DataUnderstandingEngine:
    """
    Intelligent data understanding engine.
    
    Analyzes data structure and semantics before running ML algorithms.
    """
    
    # Common ID column patterns
    ID_PATTERNS = [
        "id", "uuid", "guid", "key", "_id", "pk", 
        "serial", "code", "number", "no", "num"
    ]
    
    # Common time column patterns
    TIME_PATTERNS = [
        "date", "time", "datetime", "timestamp", "created", "updated",
        "start", "end", "when", "at", "on", "day", "month", "year"
    ]
    
    # Common grouping column patterns
    GROUP_PATTERNS = [
        "type", "category", "class", "group", "segment", "region",
        "country", "city", "state", "department", "channel", "source",
        "route", "flight", "airline", "origin", "destination", "status"
    ]
    
    def __init__(self):
        from core.logging_config import data_logger
        self.logger = data_logger
    
    def understand(self, df: pl.DataFrame) -> DataUnderstanding:
        """
        Analyze and understand the dataset structure.
        
        Returns a DataUnderstanding object with recommendations.
        """
        self.logger.info("=== DATA UNDERSTANDING STARTED ===")
        
        # Basic info
        row_count = len(df)
        column_count = len(df.columns)
        self.logger.info(f"Analyzing {row_count:,} rows x {column_count} columns")
        
        # Identify column types
        self.logger.debug("Identifying time columns...")
        time_columns = self._identify_time_columns(df)
        self.logger.debug(f"Time columns: {[t.name for t in time_columns]}")
        
        self.logger.debug("Identifying grouping columns...")
        grouping_columns = self._identify_grouping_columns(df)
        self.logger.debug(f"Grouping columns: {[g.name for g in grouping_columns]}")
        
        self.logger.debug("Identifying ID columns...")
        id_columns = self._identify_id_columns(df)
        self.logger.debug(f"ID columns: {id_columns}")  # id_columns is list[str]
        
        self.logger.debug("Identifying measure columns...")
        measure_columns = self._identify_measure_columns(df, id_columns)
        self.logger.debug(f"Measure columns: {[m.name for m in measure_columns]}")
        
        # Determine dataset type
        self.logger.debug("Determining dataset type...")
        dataset_type = self._determine_dataset_type(
            df, time_columns, grouping_columns
        )
        
        # Determine analysis scope
        primary_grouper, recommended_scope = self._determine_analysis_scope(
            df, grouping_columns
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            df, 
            dataset_type, 
            time_columns, 
            grouping_columns,
            measure_columns,
            primary_grouper,
            recommended_scope
        )
        
        # Check data quality
        missing_severity, quality_issues = self._assess_data_quality(df)
        
        # SEMANTIC UNDERSTANDING - What is this data about?
        self.logger.info("Inferring semantic understanding...")
        business_domain = self._infer_business_domain(df, measure_columns, grouping_columns)
        self.logger.success(f"Business domain: {business_domain}")
        
        data_story = self._generate_data_story(
            df, dataset_type, measure_columns, grouping_columns, time_columns, business_domain
        )
        self.logger.debug(f"Data story: {data_story[:80]}...")
        
        key_questions = self._generate_key_questions(
            business_domain, measure_columns, grouping_columns, time_columns
        )
        self.logger.debug(f"Generated {len(key_questions)} key questions")
        
        story_columns = self._identify_story_columns(
            measure_columns, grouping_columns, time_columns, business_domain
        )
        self.logger.debug(f"Identified {len(story_columns)} story columns")
        
        self.logger.success(f"=== DATA UNDERSTANDING COMPLETE: {dataset_type.value} ===")
        
        return DataUnderstanding(
            dataset_type=dataset_type,
            row_count=row_count,
            column_count=column_count,
            time_columns=time_columns,
            grouping_columns=grouping_columns,
            measure_columns=measure_columns,
            id_columns=id_columns,
            recommended_scope=recommended_scope,
            primary_grouper=primary_grouper,
            analysis_recommendations=recommendations,
            missing_data_severity=missing_severity,
            data_quality_issues=quality_issues,
            # Semantic understanding
            business_domain=business_domain,
            data_story=data_story,
            key_questions=key_questions,
            story_columns=story_columns,
        )
    
    def _identify_time_columns(self, df: pl.DataFrame) -> list[TimeColumn]:
        """Identify columns that represent time."""
        time_cols = []
        
        for col in df.columns:
            dtype = df[col].dtype
            col_lower = col.lower()
            
            # Check if datetime type - ONLY accept actual datetime types
            is_datetime = dtype in (
                pl.Datetime, pl.Date, pl.Time, 
                pl.Datetime("ns"), pl.Datetime("us"), pl.Datetime("ms")
            )
            
            # For string columns, we must verify they contain parseable dates
            # Don't just trust the column name - "Region" contains "on" but isn't a date!
            if not is_datetime and dtype in (pl.Utf8, pl.String):
                # Only consider if name STRONGLY suggests time
                strong_time_patterns = ["date", "datetime", "timestamp", "time"]
                name_strongly_suggests_time = any(
                    pattern in col_lower for pattern in strong_time_patterns
                )
                
                if name_strongly_suggests_time:
                    # Try to parse a sample to verify it's actually a date
                    sample = df[col].drop_nulls().head(10).to_list()
                    if sample and self._looks_like_dates(sample):
                        is_datetime = True  # Treat as datetime column
            
            if is_datetime:
                col_data = df[col].drop_nulls()
                
                if len(col_data) > 1:
                    # Check if sorted
                    try:
                        is_sorted = col_data.is_sorted()
                    except:
                        is_sorted = False
                    
                    has_gaps = False
                    frequency = None
                    
                    # Date range
                    try:
                        min_val = str(col_data.min())
                        max_val = str(col_data.max())
                        date_range = (min_val, max_val)
                    except:
                        date_range = None
                    
                    time_cols.append(TimeColumn(
                        name=col,
                        is_sorted=is_sorted,
                        has_gaps=has_gaps,
                        frequency=frequency,
                        date_range=date_range,
                    ))
        
        return time_cols
    
    def _looks_like_dates(self, samples: list) -> bool:
        """Check if sample values look like dates."""
        import re
        
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{4}-\d{2}-\d{2}',          # YYYY-MM-DD
            r'\d{1,2}-\d{1,2}-\d{2,4}',    # DD-MM-YYYY
            r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}',  # Month DD, YYYY
            r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}',    # DD Month YYYY
        ]
        
        matches = 0
        for sample in samples[:5]:
            if sample is None:
                continue
            sample_str = str(sample)
            for pattern in date_patterns:
                if re.match(pattern, sample_str):
                    matches += 1
                    break
        
        # At least 60% should match date patterns
        return matches >= len(samples[:5]) * 0.6
    
    def _identify_grouping_columns(self, df: pl.DataFrame) -> list[GroupingColumn]:
        """Identify columns useful for grouping/segmentation."""
        grouping_cols = []
        row_count = len(df)
        
        for col in df.columns:
            dtype = df[col].dtype
            col_lower = col.lower()
            unique_count = df[col].n_unique()
            
            # Skip if too few or too many unique values
            if unique_count < 2 or unique_count > row_count * 0.5:
                continue
            
            # Categorical or string with reasonable cardinality
            is_categorical = dtype in (pl.Utf8, pl.String, pl.Categorical)
            name_suggests_group = any(
                pattern in col_lower 
                for pattern in self.GROUP_PATTERNS
            )
            
            # Good grouping column: categorical with 2-100 unique values
            # or name suggests grouping with reasonable cardinality
            is_good_grouper = (
                (is_categorical and 2 <= unique_count <= 100) or
                (name_suggests_group and unique_count <= 500) or
                (not is_categorical and 2 <= unique_count <= 20)
            )
            
            if is_good_grouper:
                # Get value distribution
                value_counts = df[col].value_counts().sort("count", descending=True)
                top_values = {}
                coverage = 0
                
                for i, row in enumerate(value_counts.head(10).iter_rows(named=True)):
                    val_col = [c for c in value_counts.columns if c != "count"][0]
                    val = str(row[val_col])
                    count = row["count"]
                    top_values[val] = count
                    if i < 5:  # Top 5 coverage
                        coverage += count
                
                coverage_pct = (coverage / row_count) * 100 if row_count > 0 else 0
                
                # Primary grouper if covers most of data with few groups
                is_primary = (
                    coverage_pct > 80 and 
                    unique_count <= 20 and
                    name_suggests_group
                )
                
                grouping_cols.append(GroupingColumn(
                    name=col,
                    unique_values=unique_count,
                    value_distribution=top_values,
                    is_primary=is_primary,
                    coverage=coverage_pct,
                ))
        
        # Sort by likelihood of being a good grouper
        grouping_cols.sort(key=lambda x: (x.is_primary, x.coverage), reverse=True)
        
        return grouping_cols
    
    def _identify_id_columns(self, df: pl.DataFrame) -> list[str]:
        """Identify columns that are just IDs (not useful for analysis)."""
        id_cols = []
        row_count = len(df)
        
        for col in df.columns:
            col_lower = col.lower()
            unique_count = df[col].n_unique()
            
            # Name suggests ID
            name_suggests_id = any(
                col_lower.endswith(pattern) or col_lower.startswith(pattern)
                for pattern in self.ID_PATTERNS
            )
            
            # High cardinality (almost all unique)
            high_cardinality = unique_count > row_count * 0.9
            
            if name_suggests_id or high_cardinality:
                id_cols.append(col)
        
        return id_cols
    
    def _identify_measure_columns(
        self, 
        df: pl.DataFrame, 
        id_columns: list[str]
    ) -> list[MeasureColumn]:
        """Identify numeric columns that are measures (not IDs)."""
        measures = []
        
        for col in df.columns:
            if col in id_columns:
                continue
            
            dtype = df[col].dtype
            col_lower = col.lower()
            
            if dtype not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                           pl.Float32, pl.Float64):
                continue
            
            # Determine analysis type
            if any(x in col_lower for x in ["price", "cost", "amount", "revenue", "sales"]):
                analysis_type = "currency"
                aggregatable = True
            elif any(x in col_lower for x in ["count", "quantity", "total", "num"]):
                analysis_type = "count"
                aggregatable = True
            elif any(x in col_lower for x in ["rate", "ratio", "percent", "pct", "%"]):
                analysis_type = "ratio"
                aggregatable = False  # Ratios shouldn't be summed
            else:
                analysis_type = "numeric"
                aggregatable = True
            
            measures.append(MeasureColumn(
                name=col,
                analysis_type=analysis_type,
                aggregatable=aggregatable,
            ))
        
        return measures
    
    def _determine_dataset_type(
        self,
        df: pl.DataFrame,
        time_columns: list[TimeColumn],
        grouping_columns: list[GroupingColumn],
    ) -> DatasetType:
        """Determine the overall type of dataset."""
        has_time = len(time_columns) > 0
        has_groups = len(grouping_columns) > 0
        
        # Check if time column is sorted (time series characteristic)
        time_is_sorted = any(tc.is_sorted for tc in time_columns)
        
        if has_time and time_is_sorted and has_groups:
            return DatasetType.PANEL  # Both time and cross-sectional
        elif has_time and time_is_sorted:
            return DatasetType.TIME_SERIES
        elif has_groups:
            return DatasetType.HIERARCHICAL
        elif has_time:
            return DatasetType.TRANSACTIONAL
        else:
            return DatasetType.CROSS_SECTIONAL
    
    def _determine_analysis_scope(
        self,
        df: pl.DataFrame,
        grouping_columns: list[GroupingColumn],
    ) -> tuple[Optional[str], AnalysisScope]:
        """Determine whether to analyze globally or by groups."""
        
        if not grouping_columns:
            return None, AnalysisScope.GLOBAL
        
        # Find primary grouper
        primary_candidates = [g for g in grouping_columns if g.is_primary]
        
        if primary_candidates:
            primary = primary_candidates[0].name
            
            # If there's a clear primary grouper with good coverage, analyze by group
            if primary_candidates[0].coverage > 90:
                return primary, AnalysisScope.BY_GROUP
            else:
                return primary, AnalysisScope.BOTH
        
        # If we have grouping columns but no clear primary, suggest both
        best_grouper = grouping_columns[0].name if grouping_columns else None
        
        if best_grouper and grouping_columns[0].unique_values <= 10:
            return best_grouper, AnalysisScope.BOTH
        
        return None, AnalysisScope.GLOBAL
    
    def _generate_recommendations(
        self,
        df: pl.DataFrame,
        dataset_type: DatasetType,
        time_columns: list[TimeColumn],
        grouping_columns: list[GroupingColumn],
        measure_columns: list[MeasureColumn],
        primary_grouper: Optional[str],
        recommended_scope: AnalysisScope,
    ) -> list[AnalysisRecommendation]:
        """Generate analysis recommendations based on data understanding."""
        recommendations = []
        
        measure_names = [m.name for m in measure_columns]
        
        # Trend analysis - only for time series
        if dataset_type in (DatasetType.TIME_SERIES, DatasetType.PANEL):
            recommendations.append(AnalysisRecommendation(
                analysis_name="trend_analysis",
                applicable=True,
                reason="Dataset has time dimension - analyze trends over time",
                scope=recommended_scope,
                target_columns=measure_names,
                group_by=primary_grouper if recommended_scope != AnalysisScope.GLOBAL else None,
                priority=9,
            ))
            
            # Seasonality for time series
            recommendations.append(AnalysisRecommendation(
                analysis_name="seasonality_analysis",
                applicable=True,
                reason="Time series data may have seasonal patterns",
                scope=recommended_scope,
                target_columns=measure_names,
                group_by=primary_grouper if recommended_scope != AnalysisScope.GLOBAL else None,
                priority=7,
            ))
        else:
            recommendations.append(AnalysisRecommendation(
                analysis_name="trend_analysis",
                applicable=False,
                reason="No clear time dimension for trend analysis",
                scope=AnalysisScope.GLOBAL,
                target_columns=[],
                priority=1,
            ))
        
        # Outlier detection - always applicable if we have measures
        if measure_columns:
            recommendations.append(AnalysisRecommendation(
                analysis_name="outlier_detection",
                applicable=True,
                reason="Numeric columns available for outlier detection",
                scope=recommended_scope,
                target_columns=measure_names,
                group_by=primary_grouper if recommended_scope == AnalysisScope.BY_GROUP else None,
                priority=8,
            ))
        
        # Correlation - need multiple numeric columns
        if len(measure_columns) >= 2:
            recommendations.append(AnalysisRecommendation(
                analysis_name="correlation_analysis",
                applicable=True,
                reason=f"Multiple numeric columns ({len(measure_columns)}) available for correlation",
                scope=AnalysisScope.GLOBAL,  # Correlation usually global
                target_columns=measure_names,
                priority=6,
            ))
        else:
            recommendations.append(AnalysisRecommendation(
                analysis_name="correlation_analysis",
                applicable=False,
                reason="Need at least 2 numeric columns for correlation",
                scope=AnalysisScope.GLOBAL,
                target_columns=[],
                priority=1,
            ))
        
        # Group comparison - if we have groups
        if grouping_columns and measure_columns:
            recommendations.append(AnalysisRecommendation(
                analysis_name="group_comparison",
                applicable=True,
                reason=f"Compare measures across {grouping_columns[0].name}",
                scope=AnalysisScope.BY_GROUP,
                target_columns=measure_names,
                group_by=grouping_columns[0].name,
                priority=8,
            ))
        
        # Distribution analysis
        if measure_columns:
            recommendations.append(AnalysisRecommendation(
                analysis_name="distribution_analysis",
                applicable=True,
                reason="Analyze shape and spread of numeric data",
                scope=recommended_scope,
                target_columns=measure_names,
                group_by=primary_grouper if recommended_scope != AnalysisScope.GLOBAL else None,
                priority=5,
            ))
        
        # Key influencers - if we have target and features
        if measure_columns and (grouping_columns or len(measure_columns) > 1):
            recommendations.append(AnalysisRecommendation(
                analysis_name="key_influencers",
                applicable=True,
                reason=f"Find factors that influence {measure_columns[0].name}",
                scope=AnalysisScope.GLOBAL,
                target_columns=[measure_columns[0].name],
                priority=7,
            ))
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations
    
    def _assess_data_quality(
        self,
        df: pl.DataFrame,
    ) -> tuple[str, list[str]]:
        """Assess data quality issues."""
        issues = []
        total_cells = len(df) * len(df.columns)
        total_nulls = sum(df[col].null_count() for col in df.columns)
        null_pct = (total_nulls / total_cells) * 100 if total_cells > 0 else 0
        
        if null_pct > 20:
            severity = "high"
            issues.append(f"High missing data: {null_pct:.1f}% of values are null")
        elif null_pct > 5:
            severity = "medium"
            issues.append(f"Moderate missing data: {null_pct:.1f}% of values are null")
        else:
            severity = "low"
        
        # Check for columns with mostly nulls
        for col in df.columns:
            col_null_pct = (df[col].null_count() / len(df)) * 100
            if col_null_pct > 50:
                issues.append(f"Column '{col}' is {col_null_pct:.0f}% null")
        
        # Check for constant columns
        for col in df.columns:
            if df[col].n_unique() == 1:
                issues.append(f"Column '{col}' has only one unique value")
        
        return severity, issues
    
    # =============== SEMANTIC UNDERSTANDING METHODS ===============
    
    # Business domain detection patterns
    DOMAIN_PATTERNS = {
        "sales": ["sales", "revenue", "order", "customer", "product", "item", "price", "cost", "profit", "quantity", "units sold"],
        "finance": ["budget", "expense", "income", "balance", "account", "transaction", "payment", "invoice", "tax", "fiscal"],
        "hr": ["employee", "salary", "department", "position", "hire", "tenure", "performance", "attendance", "leave"],
        "marketing": ["campaign", "leads", "conversion", "clicks", "impressions", "engagement", "channel", "audience", "roi"],
        "operations": ["inventory", "shipment", "delivery", "warehouse", "supply", "logistics", "production", "manufacturing"],
        "healthcare": ["patient", "diagnosis", "treatment", "medical", "hospital", "prescription", "health"],
        "ecommerce": ["cart", "checkout", "shipping", "sku", "catalog", "vendor", "merchant"],
    }
    
    def _infer_business_domain(
        self,
        df: pl.DataFrame,
        measures: list[MeasureColumn],
        groups: list[GroupingColumn],
    ) -> str:
        """Infer the business domain from column names and values."""
        all_columns = [c.lower() for c in df.columns]
        all_text = " ".join(all_columns)
        
        # Also check measure and group names
        for m in measures:
            all_text += f" {m.name.lower()}"
        for g in groups:
            all_text += f" {g.name.lower()}"
        
        # Score each domain
        domain_scores = {}
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            score = sum(1 for p in patterns if p in all_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return "general"
    
    def _generate_data_story(
        self,
        df: pl.DataFrame,
        dataset_type: DatasetType,
        measures: list[MeasureColumn],
        groups: list[GroupingColumn],
        time_cols: list[TimeColumn],
        domain: str,
    ) -> str:
        """Generate a one-sentence story about what this data represents."""
        row_count = len(df)
        
        # Build story components
        measure_names = [m.name for m in measures[:3]]
        group_names = [g.name for g in groups[:3]]
        
        # Domain-specific story starters
        domain_starters = {
            "sales": f"This is a sales dataset with {row_count:,} records tracking",
            "finance": f"This is a financial dataset with {row_count:,} records containing",
            "hr": f"This is an HR dataset with {row_count:,} employee records showing",
            "marketing": f"This is a marketing dataset with {row_count:,} records measuring",
            "operations": f"This is an operations dataset with {row_count:,} records tracking",
            "general": f"This dataset contains {row_count:,} records with",
        }
        
        story = domain_starters.get(domain, domain_starters["general"])
        
        # Add measures
        if measure_names:
            story += f" {', '.join(measure_names)}"
        
        # Add dimensions
        if group_names:
            story += f" across {', '.join(group_names)}"
        
        # Add time context
        if time_cols:
            story += f" over time ({time_cols[0].name})"
        
        story += "."
        
        return story
    
    def _generate_key_questions(
        self,
        domain: str,
        measures: list[MeasureColumn],
        groups: list[GroupingColumn],
        time_cols: list[TimeColumn],
    ) -> list[str]:
        """Generate business questions this data can answer."""
        questions = []
        
        measure_names = [m.name for m in measures[:3]]
        group_names = [g.name for g in groups[:3]]
        
        if not measure_names:
            return ["What patterns exist in this data?"]
        
        primary_measure = measure_names[0]
        
        # Comparison questions
        for group in group_names[:2]:
            questions.append(f"Which {group} has the highest/lowest {primary_measure}?")
        
        # Trend questions
        if time_cols:
            questions.append(f"How has {primary_measure} changed over time?")
            questions.append(f"Are there any seasonal patterns in {primary_measure}?")
        
        # Distribution questions
        questions.append(f"What is the distribution of {primary_measure}?")
        
        # Relationship questions
        if len(measure_names) >= 2:
            questions.append(f"Is there a relationship between {measure_names[0]} and {measure_names[1]}?")
        
        # Outlier questions
        questions.append(f"Are there any unusual values or outliers in {primary_measure}?")
        
        return questions[:5]  # Limit to 5 questions
    
    def _identify_story_columns(
        self,
        measures: list[MeasureColumn],
        groups: list[GroupingColumn],
        time_cols: list[TimeColumn],
        domain: str,
    ) -> list[StoryColumn]:
        """Identify the most important columns for storytelling."""
        story_cols = []
        
        # Primary measure - most important for the story
        if measures:
            # Find the most "important" measure based on domain
            primary_patterns = {
                "sales": ["revenue", "sales", "profit", "total"],
                "finance": ["amount", "balance", "income", "budget"],
                "hr": ["salary", "headcount", "performance"],
                "marketing": ["conversion", "leads", "roi", "engagement"],
            }
            
            domain_patterns = primary_patterns.get(domain, ["total", "amount", "value"])
            
            # Score measures
            best_primary = None
            best_score = -1
            for m in measures:
                score = sum(1 for p in domain_patterns if p in m.name.lower())
                if score > best_score or best_primary is None:
                    best_primary = m
                    best_score = score
            
            if best_primary:
                story_cols.append(StoryColumn(
                    name=best_primary.name,
                    role="primary_measure",
                    business_meaning=self._infer_business_meaning(best_primary.name, domain),
                    importance=1.0,
                ))
            
            # Secondary measures
            for m in measures:
                if m.name != best_primary.name and len(story_cols) < 5:
                    story_cols.append(StoryColumn(
                        name=m.name,
                        role="secondary_measure",
                        business_meaning=self._infer_business_meaning(m.name, domain),
                        importance=0.7,
                    ))
        
        # Key dimensions
        for g in groups[:2]:
            story_cols.append(StoryColumn(
                name=g.name,
                role="key_dimension",
                business_meaning=self._infer_business_meaning(g.name, domain),
                importance=0.8,
            ))
        
        # Time axis
        if time_cols:
            story_cols.append(StoryColumn(
                name=time_cols[0].name,
                role="time_axis",
                business_meaning="Time dimension for trend analysis",
                importance=0.9,
            ))
        
        # Sort by importance
        story_cols.sort(key=lambda x: x.importance, reverse=True)
        
        return story_cols[:6]  # Limit to top 6
    
    def _infer_business_meaning(self, column_name: str, domain: str) -> str:
        """Infer what a column means in business terms."""
        name_lower = column_name.lower()
        
        meanings = {
            # Sales
            "revenue": "Money earned from sales",
            "total revenue": "Total money earned from sales",
            "profit": "Revenue minus costs",
            "total profit": "Total earnings after costs",
            "units sold": "Number of items sold",
            "unit price": "Price per single item",
            "unit cost": "Cost to produce/acquire one item",
            "total cost": "Total cost of goods/operations",
            "order": "Customer purchase transaction",
            "sales channel": "How the sale was made (online/offline)",
            "item type": "Category of product",
            "region": "Geographic area",
            "country": "Country location",
            "order priority": "Urgency level of the order",
            "order date": "When the order was placed",
            "ship date": "When the order was shipped",
            # General
            "quantity": "Number of items",
            "amount": "Monetary value",
            "category": "Classification group",
            "type": "Type or classification",
            "status": "Current state or condition",
            "date": "Point in time",
        }
        
        # Check exact matches first
        for key, meaning in meanings.items():
            if key == name_lower or key.replace(" ", "_") == name_lower:
                return meaning
        
        # Check partial matches
        for key, meaning in meanings.items():
            if key in name_lower:
                return meaning
        
        return f"Field for {column_name}"


# Global instance
data_understanding_engine = DataUnderstandingEngine()
