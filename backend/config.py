"""
Smart Report Insights Engine - Configuration

High-performance configuration using Pydantic Settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Ollama LLM configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")
    
    base_url: str = Field(
        default="http://192.168.29.179:9000",
        description="Ollama API base URL"
    )
    model: str = Field(
        default="llama3.2:latest",
        description="Default model for analysis"
    )
    timeout: float = Field(
        default=120.0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )


class CacheSettings(BaseSettings):
    """Caching configuration."""
    
    model_config = SettingsConfigDict(env_prefix="CACHE_")
    
    enabled: bool = Field(default=True, description="Enable caching")
    max_size: int = Field(default=128, description="LRU cache max size")
    ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")


class AnalysisSettings(BaseSettings):
    """Analysis engine configuration."""
    
    model_config = SettingsConfigDict(env_prefix="ANALYSIS_")
    
    # Quick Insights
    quick_insights_top_n: int = Field(
        default=10,
        description="Number of top insights to return"
    )
    
    # Outlier Detection
    outlier_iqr_multiplier: float = Field(
        default=1.5,
        description="IQR multiplier for outlier detection"
    )
    outlier_zscore_threshold: float = Field(
        default=3.0,
        description="Z-score threshold for outliers"
    )
    
    # Correlation
    correlation_significance_level: float = Field(
        default=0.05,
        description="P-value threshold for significance"
    )
    min_correlation_strength: float = Field(
        default=0.3,
        description="Minimum correlation to report"
    )
    
    # Trends
    trend_min_points: int = Field(
        default=5,
        description="Minimum data points for trend analysis"
    )
    
    # Key Influencers
    shap_max_samples: int = Field(
        default=1000,
        description="Max samples for SHAP analysis"
    )
    feature_importance_top_n: int = Field(
        default=10,
        description="Top N features to report"
    )
    
    # Performance
    max_workers: int = Field(
        default=4,
        description="Max parallel workers for analysis"
    )
    chunk_size: int = Field(
        default=100000,
        description="Chunk size for streaming large files"
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application
    app_name: str = "Smart Report Insights Engine"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # File Upload
    max_file_size_mb: int = Field(
        default=500,
        description="Maximum file size in MB"
    )
    upload_dir: str = Field(
        default="./uploads",
        description="Directory for uploaded files"
    )
    
    # Session
    session_ttl_hours: int = Field(
        default=24,
        description="Session time-to-live in hours"
    )
    
    # Nested settings
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
