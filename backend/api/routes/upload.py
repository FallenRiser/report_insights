"""
Upload API Routes

Endpoints for CSV file upload and session management.
"""

import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas.responses import DataProfile, SessionInfo, UploadResponse
from config import get_settings
from core.cache import session_store
from core.csv_parser import csv_parser
from core.data_profiler import data_profiler


router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload a CSV file for analysis.
    
    Creates a new session and returns basic data profile.
    """
    settings = get_settings()
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB"
        )
    
    try:
        # Parse CSV
        df = csv_parser.parse_bytes(content, file.filename)
        
        # Generate session ID
        session_id = csv_parser.generate_session_id(file.filename)
        
        # Save to disk (Parquet for fast loading)
        csv_parser.save_dataframe(df, session_id)
        
        # Profile the data
        profile = data_profiler.profile(df)
        
        # Store session metadata
        session_store.create(session_id, {
            "filename": file.filename,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns,
            "status": "ready",
            "file_size_mb": file_size_mb,
        })
        
        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            row_count=len(df),
            column_count=len(df.columns),
            columns=df.columns,
            profile=profile,
            message=f"Successfully uploaded and processed {file.filename}",
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    """Get session information."""
    
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionInfo(
        session_id=session_id,
        filename=session.get("filename", "unknown"),
        created_at=datetime.fromtimestamp(session.get("created_at", 0)),
        row_count=session.get("row_count", 0),
        column_count=session.get("column_count", 0),
        columns=session.get("columns", []),
        status=session.get("status", "unknown"),
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session and its data."""
    
    # Delete session metadata
    if not session_store.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete data file
    csv_parser.delete_session(session_id)
    
    return {"message": f"Session {session_id} deleted successfully"}


@router.get("/sessions")
async def list_sessions() -> dict:
    """List all active sessions."""
    
    session_ids = session_store.list_sessions()
    sessions = []
    
    for sid in session_ids:
        session = session_store.get(sid)
        if session:
            sessions.append({
                "session_id": sid,
                "filename": session.get("filename"),
                "row_count": session.get("row_count"),
                "status": session.get("status"),
            })
    
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/sessions/{session_id}/profile", response_model=DataProfile)
async def get_profile(session_id: str) -> DataProfile:
    """Get detailed data profile for a session."""
    
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        df = csv_parser.load_dataframe(session_id)
        profile = data_profiler.profile(df)
        return profile
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session data not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
async def clear_cache() -> dict:
    """
    Clear all cached data and sessions.
    
    Use this if you see stale data or want to force fresh analysis.
    """
    from core.cache import analysis_cache, session_store
    from pathlib import Path
    
    # Clear analysis cache
    analysis_cache.clear()
    
    # Clear all sessions
    sessions_cleared = session_store.clear_all()
    
    # Also clear parquet files
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    files_deleted = 0
    
    if upload_dir.exists():
        for file in upload_dir.glob("*.parquet"):
            try:
                file.unlink()
                files_deleted += 1
            except Exception:
                pass
    
    return {
        "message": "Cache cleared successfully",
        "sessions_cleared": sessions_cleared,
        "files_deleted": files_deleted,
        "analysis_cache": "cleared",
    }


@router.delete("/clear-all")
async def clear_all_data() -> dict:
    """
    Alias for /clear-cache. Clear all cached data, sessions, and files.
    """
    return await clear_cache()
