"""
Smart Report Insights Engine - Main Application

High-performance FastAPI server for data analysis.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from api.routes import upload, quick_insights, report_insights, chat


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    
    # Startup
    os.makedirs(settings.upload_dir, exist_ok=True)
    print(f"🚀 {settings.app_name} v{settings.app_version} starting...")
    print(f"📁 Upload directory: {settings.upload_dir}")
    print(f"🤖 Ollama model: {settings.ollama.model}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="High-performance report insights engine with Quick Insights, Report Insights, and Conversational Analysis",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
    app.include_router(quick_insights.router, prefix="/api/v1", tags=["Quick Insights"])
    app.include_router(report_insights.router, prefix="/api/v1", tags=["Report Insights"])
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version
        }
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
