from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from app.config.database import Base

# ==========================================
# SQLAlchemy ORM Model
# ==========================================
class SkinScan(Base):
    """SkinScan model representing a captured face scan session."""
    __tablename__ = "skin_scans"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    image_path = Column(String(255), nullable=False)
    status = Column(String(50), default="completed") # pending, processing, completed, failed
    image_quality_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

# ==========================================
# Pydantic Schemas (for validation)
# ==========================================
class SkinScanSchema(BaseModel):
    id: int
    user_id: int
    image_path: str
    status: str
    image_quality_score: float
    created_at: datetime

    class Config:
        from_attributes = True
