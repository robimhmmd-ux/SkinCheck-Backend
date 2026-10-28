from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, Float, Text, DateTime, ForeignKey
from app.config.database import Base

# ==========================================
# SQLAlchemy ORM Model
# ==========================================
class AnalysisResult(Base):
    """AnalysisResult model representing skin condition indicators."""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    skin_scan_id = Column(Integer, ForeignKey("skin_scans.id", ondelete="CASCADE"), nullable=False)
    acne_score = Column(Float, nullable=False, default=0.0)
    dry_skin_score = Column(Float, nullable=True, default=0.0)
    oily_skin_score = Column(Float, nullable=True, default=0.0)
    dark_spots_score = Column(Float, nullable=True, default=0.0)
    wrinkles_score = Column(Float, nullable=True, default=0.0)
    redness_score = Column(Float, nullable=True, default=0.0)
    texture_score = Column(Float, nullable=True, default=0.0)
    hyperpigmentation_score = Column(Float, nullable=True, default=0.0)
    overall_score = Column(Float, nullable=False, default=0.0)
    ai_response = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def get_condition_matrix(self):
        return {
            "acne": self.acne_score or 0.0,
            "dry": self.dry_skin_score or 0.0,
            "oily": self.oily_skin_score or 0.0,
            "dark_spots": self.dark_spots_score or 0.0,
            "wrinkles": self.wrinkles_score or 0.0
        }

# ==========================================
# Pydantic Schemas (for validation)
# ==========================================
class AnalysisResultSchema(BaseModel):
    id: int
    skin_scan_id: int
    acne_score: float
    dry_skin_score: Optional[float] = 0.0
    oily_skin_score: Optional[float] = 0.0
    dark_spots_score: Optional[float] = 0.0
    wrinkles_score: Optional[float] = 0.0
    redness_score: Optional[float] = 0.0
    texture_score: Optional[float] = 0.0
    hyperpigmentation_score: Optional[float] = 0.0
    overall_score: float
    ai_response: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
