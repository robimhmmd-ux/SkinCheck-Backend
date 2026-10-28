from fastapi import APIRouter
from app.config import settings

router = APIRouter(prefix="/api/v1")

@router.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint to verify backend system status."""
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "version": "1.0.0-foundation",
        "features": {
            "web_camera_webrtc": "ready",
            "mvc_architecture": "active",
            "ai_integration": "pending_next_phase"
        }
    }
