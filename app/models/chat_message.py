from datetime import datetime
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from app.config.database import Base

# ==========================================
# SQLAlchemy ORM Model
# ==========================================
class ChatMessage(Base):
    """ChatMessage model storing conversation memory per user."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

# ==========================================
# Pydantic Schemas (for validation)
# ==========================================
class ChatMessageSchema(BaseModel):
    id: int
    user_id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True
