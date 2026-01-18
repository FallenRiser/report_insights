"""
Report Insights API Routes

DEPRECATED: Use /quick-insights/{session_id}?include_advanced=true instead.

These endpoints will remain for backwards compatibility but redirect
to the unified quick-insights API.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from api.schemas.requests import KeyInfluencersRequest, DecompositionRequest
from api.schemas.responses import ReportInsightsResponse
from core.cache import session_store
from core.csv_parser import csv_parser
from insights.report_generator import report_insights_generator


router = APIRouter()


@router.get(
    "/report-insights/{session_id}",
    response_model=ReportInsightsResponse,
    deprecated=True,
    description="DEPRECATED: Use /quick-insights/{session_id}?include_advanced=true instead"
)
async def get_report_insights(session_id: str) -> ReportInsightsResponse:
    """
    DEPRECATED: Use /quick-insights/{session_id}?include_advanced=true instead.
    
    Generate comprehensive report insights.
    
    Auto-detects all columns and generates in-depth analysis including:
    - Data profile
    - All Power BI style insights
    - Correlations
    - Key influencers (auto-selects target)
    - Decomposition (auto-selects measure)
    
    No parameters required - everything is auto-detected.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        df = csv_parser.load_dataframe(session_id)
        
        # Auto-detect target and measure columns using data understanding
        from core.data_understanding import data_understanding_engine
        understanding = data_understanding_engine.understand(df)
        
        # Use first measure as target, if available
        target_column = understanding.measure_columns[0].name if understanding.measure_columns else None
        measure_column = understanding.measure_columns[0].name if understanding.measure_columns else None
        
        response = await report_insights_generator.generate(
            df=df,
            session_id=session_id,
            target_column=target_column,
            measure_column=measure_column,
        )
        
        return response
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session data not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@router.post("/report-insights/{session_id}/key-influencers")
async def analyze_key_influencers(
    session_id: str,
    request: KeyInfluencersRequest,
) -> dict:
    """
    Analyze key influencers for a target column.
    
    Identifies which factors most strongly influence the target.
    Similar to Power BI's Key Influencers visual.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        from analysis.key_influencers import key_influencers_analyzer
        
        df = csv_parser.load_dataframe(session_id)
        
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request.target_column}' not found in dataset"
            )
        
        result = key_influencers_analyzer.analyze(
            df=df,
            target_column=request.target_column,
            top_n=request.top_n,
        )
        
        return {
            "session_id": session_id,
            **result.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report-insights/{session_id}/decomposition")
async def analyze_decomposition(
    session_id: str,
    request: DecompositionRequest,
) -> dict:
    """
    Generate decomposition tree for a measure.
    
    Breaks down a measure across dimensions to understand contributions.
    Similar to Power BI's Decomposition Tree visual.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        from analysis.decomposition import decomposition_engine
        
        df = csv_parser.load_dataframe(session_id)
        
        if request.measure_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request.measure_column}' not found in dataset"
            )
        
        # Use provided dimensions or auto-detect
        if request.dimension_columns:
            dimensions = request.dimension_columns
        else:
            from core.data_profiler import data_profiler
            dimensions = data_profiler.get_categorical_columns(df)
        
        result = decomposition_engine.decompose(
            df=df,
            measure_column=request.measure_column,
            dimension_columns=dimensions,
            aggregation=request.aggregation,
            max_depth=request.max_depth,
        )
        
        return {
            "session_id": session_id,
            **result.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report-insights/{session_id}/statistics")
async def get_statistics(
    session_id: str,
    columns: Optional[str] = Query(default=None, description="Comma-separated column names"),
) -> dict:
    """Get detailed statistics for columns."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        from analysis.statistical import statistical_analyzer
        from core.data_profiler import data_profiler
        
        df = csv_parser.load_dataframe(session_id)
        
        if columns:
            target_cols = [c.strip() for c in columns.split(",")]
        else:
            target_cols = data_profiler.get_numeric_columns(df)
        
        statistics = {}
        distributions = {}
        
        for col in target_cols:
            if col in df.columns:
                stats = statistical_analyzer.compute_descriptive_stats(df, col)
                dist = statistical_analyzer.analyze_distribution(df, col)
                
                statistics[col] = stats.to_dict()
                distributions[col] = dist.to_dict()
        
        return {
            "session_id": session_id,
            "statistics": statistics,
            "distributions": distributions,
            "columns_analyzed": list(statistics.keys()),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report-insights/{session_id}/seasonality")
async def get_seasonality(
    session_id: str,
    column: str = Query(..., description="Column to analyze"),
) -> dict:
    """Get seasonality analysis for a column."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        from analysis.seasonality import seasonality_analyzer
        
        df = csv_parser.load_dataframe(session_id)
        
        if column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column}' not found"
            )
        
        result = seasonality_analyzer.analyze(df, column)
        
        return {
            "session_id": session_id,
            **result.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report-insights/{session_id}/patterns")
async def get_patterns(session_id: str) -> dict:
    """Get all pattern analysis for the dataset."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        from analysis.patterns import pattern_recognizer
        
        df = csv_parser.load_dataframe(session_id)
        patterns = pattern_recognizer.detect_all_patterns(df)
        
        return {
            "session_id": session_id,
            **patterns
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
