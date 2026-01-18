"""
API Response Schemas

Pydantic models for API responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, field_serializer, model_serializer


def convert_numpy(obj: Any) -> Any:
    """Convert numpy types to Python native types."""
    if obj is None:
        return None
    # Handle numpy boolean types (np.bool_ is the main one, np.bool8 was removed)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    # Handle generic numpy dtypes
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


class InsightType(str, Enum):
    """Types of insights."""
    
    TREND = "trend"
    OUTLIER = "outlier"
    CORRELATION = "correlation"
    PATTERN = "pattern"
    DISTRIBUTION = "distribution"
    SEASONALITY = "seasonality"
    KEY_INFLUENCER = "key_influencer"
    ANOMALY = "anomaly"
    SUMMARY = "summary"


class InsightSeverity(str, Enum):
    """Insight importance level."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ColumnProfile(BaseModel):
    """Profile of a single column."""
    
    name: str
    dtype: str
    count: int
    null_count: int
    null_percentage: float
    unique_count: int
    
    # Numeric columns
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Categorical columns
    top_values: Optional[list[dict[str, Any]]] = None
    
    # Datetime columns
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[int] = None


class DataProfile(BaseModel):
    """Complete data profile."""
    
    row_count: int
    column_count: int
    memory_usage_mb: float
    columns: list[ColumnProfile]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    text_columns: list[str]


class Insight(BaseModel):
    """A single insight."""
    
    id: str = Field(..., description="Unique insight ID")
    type: InsightType
    severity: InsightSeverity
    title: str = Field(..., description="Brief insight title")
    description: str = Field(..., description="Detailed description")
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    
    # Context
    columns: list[str] = Field(default=[], description="Related columns")
    metrics: dict[str, Any] = Field(default={}, description="Supporting metrics")
    
    # Visualization hint
    chart_type: Optional[str] = None
    chart_data: Optional[dict[str, Any]] = None
    
    @field_serializer('metrics', 'chart_data')
    @classmethod
    def serialize_numpy_fields(cls, v: Any) -> Any:
        """Convert numpy types to Python native types."""
        return convert_numpy(v)


class TrendInsight(Insight):
    """Trend-specific insight."""
    
    direction: str = Field(..., pattern="^(increasing|decreasing|stable)$")
    slope: float
    r_squared: float
    change_points: list[dict[str, Any]] = []


class OutlierInsight(Insight):
    """Outlier-specific insight."""
    
    outlier_count: int
    outlier_percentage: float
    outlier_indices: list[int] = []
    outlier_values: list[Any] = []
    method: str


class CorrelationInsight(Insight):
    """Correlation-specific insight."""
    
    column1: str
    column2: str
    correlation: float
    p_value: float
    method: str


class KeyInfluencer(BaseModel):
    """A single key influencer."""
    
    feature: str
    importance: float
    direction: str  # positive/negative
    description: str
    shap_value: Optional[float] = None


class KeyInfluencersResult(BaseModel):
    """Key influencers analysis result."""
    
    target: str
    influencers: list[KeyInfluencer]
    model_score: float
    segments: list[dict[str, Any]] = []


class DecompositionNode(BaseModel):
    """Node in decomposition tree."""
    
    id: str
    label: str
    value: float
    percentage: float
    parent_id: Optional[str] = None
    children: list["DecompositionNode"] = []
    depth: int


class DecompositionResult(BaseModel):
    """Decomposition tree result."""
    
    measure: str
    total_value: float
    root: DecompositionNode
    suggested_splits: list[str] = []


class PowerBIInsightModel(BaseModel):
    """Power BI style insight - concise, actionable, with chart data."""
    
    insight_type: str = Field(
        ..., 
        description="Type: high_value, majority, low_variance, trend, outlier, correlation, change_point, seasonality, steady_share, time_series_outlier"
    )
    measure: str = Field(..., description="The measure being analyzed")
    dimension: Optional[str] = Field(None, description="The dimension for grouping")
    statement: str = Field(..., description="One-sentence insight")
    chart_type: str = Field(..., description="Chart type: bar, line, scatter, pie")
    chart_data: dict[str, Any] = Field(default={}, description="Data for rendering the chart")
    score: float = Field(default=0.5, ge=0, le=1)
    
    @field_serializer('chart_data')
    @classmethod
    def serialize_chart_data(cls, v: Any) -> Any:
        return convert_numpy(v)


class DatasetSummary(BaseModel):
    """Brief summary of the dataset."""
    
    row_count: int
    column_count: int
    dataset_type: str  # e.g., "hierarchical", "time_series", "transactional"
    primary_grouper: Optional[str] = None
    measures: list[str] = []
    dimensions: list[str] = []
    
    # Semantic understanding - What is this data about?
    business_domain: str = "general"  # e.g., "sales", "finance", "hr"
    data_story: str = ""  # One-sentence description of the data
    key_questions: list[str] = []  # Business questions this data can answer


class StoryColumnModel(BaseModel):
    """Column important for storytelling."""
    
    name: str
    role: str  # "primary_measure", "secondary_measure", "key_dimension", "time_axis"
    business_meaning: str
    importance: float


class QuickInsightsResponse(BaseModel):
    """
    Clean quick insights response.
    
    First understands the data, then returns Power BI style insights.
    """
    
    session_id: str
    generated_at: datetime
    processing_time_ms: float
    
    # STEP 1: Data Understanding - What is this data about?
    data_summary: DatasetSummary
    story_columns: list[StoryColumnModel] = []  # Most important columns for the story
    
    # STEP 2: The main output - Power BI style insights based on understanding
    insights: list[PowerBIInsightModel] = []
    
    # Actionable recommendations
    recommendations: list[str] = []
    
    # Optional narrative summary (LLM-generated if available)
    narrative: Optional[str] = None
    
    # ADVANCED ANALYSIS (formerly from report_insights - now unified)
    # These use dict to accept flexible output from various analyzers
    profile: Optional[DataProfile] = None  # Detailed data profile
    correlations: list[dict] = []  # All significant correlations
    key_influencers: Optional[dict] = None  # What influences the primary measure
    decomposition: Optional[dict] = None  # Breakdown of primary measure


class ReportInsightsResponse(BaseModel):
    """
    Detailed report insights - auto-detects columns, no user input needed.
    """
    
    session_id: str
    generated_at: datetime
    processing_time_ms: float
    
    # Data profile
    profile: DataProfile
    
    # All insights found
    insights: list[PowerBIInsightModel] = []
    
    # Detailed analysis results (optional)
    correlations: list[CorrelationInsight] = []
    key_influencers: Optional[KeyInfluencersResult] = None
    decomposition: Optional[DecompositionResult] = None
    
    # LLM-generated narrative
    narrative: str = ""


class SessionInfo(BaseModel):
    """Session information."""
    
    session_id: str
    filename: str
    created_at: datetime
    row_count: int
    column_count: int
    columns: list[str]
    status: str


class UploadResponse(BaseModel):
    """File upload response."""
    
    session_id: str
    filename: str
    row_count: int
    column_count: int
    columns: list[str]
    profile: DataProfile
    message: str


class ChatResponse(BaseModel):
    """Chat response."""
    
    session_id: str
    message: str
    response: str
    insights: list[Insight] = []
    suggested_questions: list[str] = []
    processing_time_ms: float


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
