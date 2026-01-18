"""
Quick Insights API Routes

Endpoints for fast, automatic data insights.
Unified API - includes both quick and advanced (report) insights.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas.responses import QuickInsightsResponse
from core.cache import session_store
from core.csv_parser import csv_parser
from insights.quick_generator import quick_insights_generator


router = APIRouter()


@router.get("/quick-insights/{session_id}", response_model=QuickInsightsResponse)
async def get_quick_insights(
    session_id: str,
    top_n: int = Query(default=20, ge=1, le=50, description="Number of top insights"),
    include_advanced: bool = Query(default=False, description="Include advanced analysis (profile, correlations, key_influencers, decomposition)"),
) -> QuickInsightsResponse:
    """
    Generate quick insights for the dataset.
    
    Fast automatic analysis similar to Power BI Quick Insights.
    Returns trends, outliers, correlations, and patterns.
    
    Set include_advanced=true to also get:
    - Full data profile
    - All significant correlations
    - Key influencers analysis
    - Decomposition tree
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        df = csv_parser.load_dataframe(session_id)
        
        response = await quick_insights_generator.generate(
            df=df,
            session_id=session_id,
            top_n=top_n,
            include_advanced=include_advanced,
        )
        
        return response
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session data not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")


@router.get("/quick-insights/{session_id}/trends")
async def get_trend_insights(
    session_id: str,
    columns: Optional[str] = Query(default=None, description="Comma-separated column names"),
) -> dict:
    """Get only trend insights."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        from analysis.trends import trend_detector
        from core.data_profiler import data_profiler
        
        df = csv_parser.load_dataframe(session_id)
        
        if columns:
            target_cols = [c.strip() for c in columns.split(",")]
        else:
            target_cols = data_profiler.get_numeric_columns(df)
        
        trends = {}
        for col in target_cols[:10]:
            if col in df.columns:
                trend = trend_detector.detect_linear_trend(df, col)
                trends[col] = trend.to_dict()
        
        return {
            "session_id": session_id,
            "trends": trends,
            "columns_analyzed": list(trends.keys()),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick-insights/{session_id}/outliers")
async def get_outlier_insights(
    session_id: str,
    columns: Optional[str] = Query(default=None, description="Comma-separated column names"),
    method: str = Query(default="consensus", pattern="^(iqr|zscore|consensus)$"),
) -> dict:
    """Get only outlier insights."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        from analysis.outliers import outlier_detector
        from core.data_profiler import data_profiler
        
        df = csv_parser.load_dataframe(session_id)
        
        if columns:
            target_cols = [c.strip() for c in columns.split(",")]
        else:
            target_cols = data_profiler.get_numeric_columns(df)
        
        outliers = {}
        for col in target_cols[:10]:
            if col in df.columns:
                if method == "iqr":
                    result = outlier_detector.detect_iqr(df, col)
                elif method == "zscore":
                    result = outlier_detector.detect_zscore(df, col)
                else:
                    result = outlier_detector.get_consensus_outliers(df, col)
                
                outliers[col] = result.to_dict()
        
        return {
            "session_id": session_id,
            "method": method,
            "outliers": outliers,
            "columns_analyzed": list(outliers.keys()),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick-insights/{session_id}/correlations")
async def get_correlation_insights(
    session_id: str,
    top_n: int = Query(default=10, ge=1, le=50),
) -> dict:
    """Get correlation insights."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        from analysis.correlations import correlation_analyzer
        from core.data_profiler import data_profiler
        
        df = csv_parser.load_dataframe(session_id)
        numeric_cols = data_profiler.get_numeric_columns(df)
        
        if len(numeric_cols) < 2:
            return {
                "session_id": session_id,
                "correlations": [],
                "message": "Need at least 2 numeric columns for correlation analysis",
            }
        
        matrix = correlation_analyzer.compute_correlation_matrix(df, numeric_cols[:15])
        
        return {
            "session_id": session_id,
            "columns": matrix.columns,
            "pearson_matrix": matrix.pearson_matrix,
            "significant_pairs": [p.to_dict() for p in matrix.significant_pairs[:top_n]],
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
