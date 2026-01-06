"""
Chat API Routes

Endpoints for conversational data analysis using Ollama LLM.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from api.schemas.requests import ChatMessage
from api.schemas.responses import ChatResponse
from core.cache import session_store
from core.csv_parser import csv_parser
from llm.conversation import conversation_manager
from llm.ollama_client import ollama_client


router = APIRouter()


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    message: ChatMessage,
) -> ChatResponse:
    """
    Send a message for conversational data analysis.
    
    Uses local Ollama LLM to answer questions about the dataset.
    Maintains conversation history for context awareness.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        df = csv_parser.load_dataframe(session_id)
        
        # Get any previous insights for context
        # In production, you might cache these
        insights = []
        
        response = await conversation_manager.chat(
            session_id=session_id,
            user_message=message.message,
            df=df,
            insights=insights,
            include_context=message.include_context,
        )
        
        return response
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session data not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing chat: {str(e)}"
        )


@router.get("/chat/{session_id}/history")
async def get_chat_history(session_id: str) -> dict:
    """Get conversation history for a session."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = conversation_manager.get_history(session_id)
    
    return {
        "session_id": session_id,
        "messages": history,
        "message_count": len(history),
    }


@router.delete("/chat/{session_id}/history")
async def clear_chat_history(session_id: str) -> dict:
    """Clear conversation history for a session."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    conversation_manager.clear_history(session_id)
    
    return {
        "session_id": session_id,
        "message": "Conversation history cleared",
    }


@router.get("/chat/{session_id}/suggest")
async def get_suggested_questions(session_id: str) -> dict:
    """Get suggested questions based on the data."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        df = csv_parser.load_dataframe(session_id)
        
        suggestions = await conversation_manager.get_suggestions(
            session_id=session_id,
            df=df,
        )
        
        return {
            "session_id": session_id,
            "suggestions": suggestions,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/status")
async def get_llm_status() -> dict:
    """Check if the LLM (Ollama) is available."""
    is_available = await ollama_client.is_available()
    models = []
    
    if is_available:
        models = await ollama_client.list_models()
    
    return {
        "ollama_available": is_available,
        "available_models": models,
        "message": "Ollama is running" if is_available else "Ollama is not available. Please start Ollama service.",
    }
