from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.config.database import Base


class OTPToken(Base):
    """
    OTP Token model — stores time-limited 6-digit verification codes.
    Used for:
      1. Email verification at registration (type='register')
      2. Account deletion confirmation (type='delete_account')
    """
    __tablename__ = "otp_tokens"

    id          = Column(Integer, primary_key=True, index=True)
    email       = Column(String(150), nullable=False, index=True)
    code        = Column(String(10), nullable=False)        # 6-digit OTP
    purpose     = Column(String(50), nullable=False)        # 'register' | 'delete_account'
    is_used     = Column(Boolean, default=False)
    expires_at  = Column(DateTime, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)

    def is_valid(self) -> bool:
        """Returns True if the OTP has not been used and has not expired."""
        return not self.is_used and datetime.utcnow() < self.expires_at
