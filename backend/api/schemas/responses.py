"""
API Response Schemas

Pydantic models for API responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


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


class QuickInsightsResponse(BaseModel):
    """Quick insights response."""
    
    session_id: str
    generated_at: datetime
    processing_time_ms: float
    insights: list[Insight]
    summary: str


class ReportInsightsResponse(BaseModel):
    """Full report insights response."""
    
    session_id: str
    generated_at: datetime
    processing_time_ms: float
    profile: DataProfile
    insights: list[Insight]
    correlations: list[CorrelationInsight]
    key_influencers: Optional[KeyInfluencersResult] = None
    decomposition: Optional[DecompositionResult] = None
    narrative: str


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
