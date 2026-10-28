from fastapi import Request, Depends
from sqlalchemy.orm import Session
from app.config.database import get_db
from app.models.user import User

class AuthException(Exception):
    """Custom exception for authentication redirects."""
    def __init__(self, redirect_url: str):
        self.redirect_url = redirect_url

def require_customer(request: Request, db: Session = Depends(get_db)):
    """Dependency to ensure the user is logged in as a customer."""
    user = request.session.get("user")
    if not user:
        raise AuthException(redirect_url="/login?error=Silakan masuk terlebih dahulu.")
    if user.get("role") != "customer":
        raise AuthException(redirect_url="/admin/dashboard?error=Akses ditolak.")
        
    # Verify user still exists in DB
    db_user = db.query(User).filter(User.id == user.get("id")).first()
    if not db_user:
        request.session.clear()
        raise AuthException(redirect_url="/login?error=Sesi tidak valid. Silakan masuk kembali.")
        
    return user

def require_admin(request: Request, db: Session = Depends(get_db)):
    """Dependency to ensure the user is logged in as an admin."""
    user = request.session.get("user")
    if not user:
        raise AuthException(redirect_url="/login?error=Silakan masuk terlebih dahulu.")
    if user.get("role") != "admin":
        raise AuthException(redirect_url="/dashboard?error=Akses ditolak.")
        
    # Verify user still exists in DB
    db_user = db.query(User).filter(User.id == user.get("id")).first()
    if not db_user:
        request.session.clear()
        raise AuthException(redirect_url="/login?error=Sesi tidak valid. Silakan masuk kembali.")
        
    return user

def get_current_user(request: Request):
    """Helper to get current user dict from session."""
    return request.session.get("user")
