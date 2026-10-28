from datetime import datetime
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from app.config.database import Base

# ==========================================
# SQLAlchemy ORM Model
# ==========================================
class ActivityLog(Base):
    """ActivityLog model representing user login/logout activities."""
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    activity = Column(String(255), nullable=False) # e.g. "Customer Login", "Admin Logout"
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# ==========================================
# Pydantic Schemas (for validation)
# ==========================================
class ActivityLogSchema(BaseModel):
    id: int
    user_id: int
    activity: str
    ip_address: str | None
    created_at: datetime

    class Config:
        from_attributes = True
