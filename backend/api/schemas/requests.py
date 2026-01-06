"""
API Request Schemas

Pydantic models for API request validation.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message request."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User message"
    )
    include_context: bool = Field(
        default=True,
        description="Include data context in LLM prompt"
    )


class KeyInfluencersRequest(BaseModel):
    """Request for key influencers analysis."""
    
    target_column: str = Field(
        ...,
        description="Target column to analyze"
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top influencers to return"
    )


class DecompositionRequest(BaseModel):
    """Request for decomposition tree analysis."""
    
    measure_column: str = Field(
        ...,
        description="Measure column to decompose"
    )
    dimension_columns: list[str] = Field(
        default=[],
        description="Dimension columns for decomposition"
    )
    aggregation: str = Field(
        default="sum",
        pattern="^(sum|mean|count|min|max)$",
        description="Aggregation function"
    )
    max_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum decomposition depth"
    )


class FilterCondition(BaseModel):
    """Filter condition for data."""
    
    column: str = Field(..., description="Column name")
    operator: str = Field(
        ...,
        pattern="^(eq|ne|gt|lt|gte|lte|in|contains)$",
        description="Filter operator"
    )
    value: Any = Field(..., description="Filter value")


class AnalysisRequest(BaseModel):
    """Base request for analysis with optional filters."""
    
    filters: list[FilterCondition] = Field(
        default=[],
        description="Optional data filters"
    )
    columns: list[str] = Field(
        default=[],
        description="Specific columns to analyze (empty = all)"
    )
