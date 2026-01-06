"""
Conversation Manager

Manages chat sessions and conversation flow for data analysis.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import polars as pl

from api.schemas.responses import Insight, ChatResponse
from core.cache import session_store
from core.data_profiler import data_profiler
from llm.ollama_client import ollama_client
from llm.prompts import (
    SYSTEM_PROMPT, 
    CONVERSATIONAL_ANALYSIS_PROMPT,
    get_suggested_questions,
)
from llm.context_builder import context_builder


@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    
    role: str  # user, assistant, system
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """A conversation session with history."""
    
    session_id: str
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


class ConversationManager:
    """
    Manages conversational data analysis sessions.
    
    Tracks conversation history and provides context-aware responses.
    """
    
    def __init__(self):
        self._conversations: dict[str, ConversationSession] = {}
        self.max_history = 10  # Max messages to keep in context
    
    def get_or_create_conversation(
        self,
        session_id: str,
    ) -> ConversationSession:
        """Get existing or create new conversation."""
        if session_id not in self._conversations:
            self._conversations[session_id] = ConversationSession(
                session_id=session_id
            )
        
        conv = self._conversations[session_id]
        conv.last_activity = time.time()
        return conv
    
    async def chat(
        self,
        session_id: str,
        user_message: str,
        df: pl.DataFrame,
        insights: list[Insight] = None,
        include_context: bool = True,
    ) -> ChatResponse:
        """
        Process a chat message and generate a response.
        
        Args:
            session_id: Data session ID
            user_message: User's question
            df: The dataset
            insights: Previously generated insights
            include_context: Whether to include data context
            
        Returns:
            ChatResponse with assistant response and suggestions
        """
        start_time = time.time()
        
        # Get conversation
        conv = self.get_or_create_conversation(session_id)
        
        # Add user message to history
        conv.messages.append(ConversationMessage(
            role="user",
            content=user_message,
        ))
        
        # Build context
        profile = data_profiler.profile(df)
        
        if include_context:
            context = context_builder.build_query_context(
                df, profile, insights or [], user_message
            )
            
            prompt = CONVERSATIONAL_ANALYSIS_PROMPT.format(**context)
        else:
            prompt = user_message
        
        # Build message history for LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Add recent conversation history
        for msg in conv.messages[-(self.max_history * 2):]:
            messages.append({
                "role": msg.role,
                "content": msg.content if msg.role == "user" else msg.content[:1000],
            })
        
        # Replace last user message with full context prompt
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = prompt
        
        # Generate response
        try:
            response_text = await ollama_client.chat(messages)
        except Exception as e:
            response_text = f"I apologize, but I encountered an error while processing your request: {str(e)}. Please ensure Ollama is running and try again."
        
        # Add assistant message to history
        conv.messages.append(ConversationMessage(
            role="assistant",
            content=response_text,
        ))
        
        # Generate suggested questions
        numeric_cols = profile.numeric_columns
        suggestions = get_suggested_questions(df.columns, numeric_cols)
        
        # Extract any insights mentioned in the response
        response_insights = self._extract_insights_from_response(
            response_text, df.columns
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            session_id=session_id,
            message=user_message,
            response=response_text,
            insights=response_insights,
            suggested_questions=suggestions,
            processing_time_ms=processing_time,
        )
    
    def _extract_insights_from_response(
        self,
        response: str,
        columns: list[str],
    ) -> list[Insight]:
        """
        Extract structured insights from LLM response.
        
        Simple keyword-based extraction for now.
        """
        insights = []
        
        # Check for mentions of specific patterns
        lower_response = response.lower()
        
        if "trend" in lower_response:
            for col in columns:
                if col.lower() in lower_response:
                    insights.append(Insight(
                        id=str(uuid4())[:8],
                        type="trend",
                        severity="medium",
                        title=f"Trend discussed: {col}",
                        description="LLM discussed trend patterns in this column.",
                        score=0.5,
                        columns=[col],
                        metrics={},
                    ))
                    break
        
        if "outlier" in lower_response or "anomal" in lower_response:
            insights.append(Insight(
                id=str(uuid4())[:8],
                type="outlier",
                severity="medium",
                title="Outliers discussed",
                description="LLM discussed outliers or anomalies in the data.",
                score=0.5,
                columns=[],
                metrics={},
            ))
        
        return insights[:3]  # Limit to 3
    
    def get_history(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Get conversation history for a session."""
        if session_id not in self._conversations:
            return []
        
        conv = self._conversations[session_id]
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
            }
            for msg in conv.messages
        ]
    
    def clear_history(self, session_id: str) -> bool:
        """Clear conversation history."""
        if session_id in self._conversations:
            self._conversations[session_id].messages = []
            return True
        return False
    
    async def get_suggestions(
        self,
        session_id: str,
        df: pl.DataFrame,
        insights: list[Insight] = None,
    ) -> list[str]:
        """Get suggested questions based on data and insights."""
        profile = data_profiler.profile(df)
        base_suggestions = get_suggested_questions(
            df.columns, 
            profile.numeric_columns
        )
        
        # Add context-aware suggestions based on insights
        if insights:
            for insight in insights[:3]:
                if insight.type == "outlier":
                    base_suggestions.append(
                        f"Tell me more about the outliers in {insight.columns[0] if insight.columns else 'the data'}"
                    )
                elif insight.type == "trend":
                    base_suggestions.append(
                        f"What's driving the trend in {insight.columns[0] if insight.columns else 'the data'}?"
                    )
        
        return base_suggestions[:8]


# Global instance
conversation_manager = ConversationManager()
