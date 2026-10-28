import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Column, Integer, String, DateTime, Text
from app.config.database import Base

# ==========================================
# SQLAlchemy ORM Model
# ==========================================
class User(Base):
    """User Model representing Customer and Admin roles."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="customer") # customer or admin
    status = Column(String(50), default="active") # active, inactive, suspended
    profile_data = Column(Text, nullable=True)    # JSON string for skin goals, concerns
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def set_profile_data(self, data: Dict[str, Any]):
        self.profile_data = json.dumps(data)

    def get_profile_data(self) -> Dict[str, Any]:
        if self.profile_data:
            return json.loads(self.profile_data)
        return {}


# ==========================================
# Pydantic Schemas (for validation)
# ==========================================
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    skin_concerns: Optional[List[str]] = []
    privacy_consent: bool = False

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserSchema(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str
    status: str
    profile_data: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
