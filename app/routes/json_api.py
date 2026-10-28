"""SkinCheck API v2 — JSON-only API for the Next.js frontend.

This is the API-only layer of the split backend. It deliberately never
returns a TemplateResponse: every endpoint returns JSONResponse, so the
Next.js frontend can consume it without any Jinja template.

Auth model (kept identical to the monolith, see AuthController):
- Session cookie ``skincheck_session`` via Starlette SessionMiddleware.
- ``request.session["user"]`` is ``{"id", "name", "email", "role"}``.
- Passwords are bcrypt-hashed via ``app.utils.auth`` (same as monolith).

OTP note:
- The monolith's registration flow is a two-step OTP (email/WhatsApp)
  flow. ``POST /api/v2/auth/register`` intentionally bypasses OTP and
  creates the account directly, mirroring the monolith's
  ``REGISTRATION_OTP_MODE=bypass`` branch. The OTP email flow stays in
  the monolith for now; if OTP is re-introduced for the API, it should
  be a separate ``/api/v2/auth/verify-otp`` endpoint, not a silent
  change to this one.

AI / scan note:
- ``POST /api/v2/scan/upload`` only validates + stores the image and
  creates a ``SkinScan`` row with ``status="processing"``. It does NOT
  run the TensorFlow model and does NOT import ``ai_service`` at all,
  so the app imports and runs without tensorflow / opencv /
  google-generativeai installed. Full AI analysis is a follow-up
  endpoint (it must run the .h5 model, which is not in this repo).

Database failure handling:
- Endpoints that touch the DB catch all DB errors and return
  ``503 {"detail": "Database unavailable"}`` as JSON, never a 500 HTML
  page. ``/api/v2/health`` and ``/api/v2/me`` (logged-out case) do not
  touch the DB at all, so they work even when the DB is down.
"""

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.config import settings
from app.config.database import get_db
from app.models.activity_log import ActivityLog
from app.models.analysis_result import AnalysisResult
from app.models.product import Product
from app.models.skin_scan import SkinScan
from app.models.user import User
from app.utils.auth import get_password_hash, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["API v2"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_dict(user: User) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "status": user.status,
    }


def _session_user(request: Request) -> Optional[dict]:
    """Session user dict, or None when logged out. No DB access."""
    user = request.session.get("user")
    return user if isinstance(user, dict) else None


def _require_session(request: Request) -> dict:
    user = _session_user(request)
    if not user:
        raise _unauthorized()
    return user


def _unauthorized():
    from fastapi import HTTPException

    return HTTPException(status_code=401, detail="Login required")


def _forbidden(detail: str = "Forbidden"):
    from fastapi import HTTPException

    return HTTPException(status_code=403, detail=detail)


def _db_unavailable() -> JSONResponse:
    return JSONResponse(
        status_code=503, content={"detail": "Database unavailable"}
    )


def _analysis_dict(a: Optional[AnalysisResult]) -> Optional[dict]:
    if not a:
        return None
    return {
        "id": a.id,
        "skin_scan_id": a.skin_scan_id,
        "acne_score": a.acne_score,
        "dry_skin_score": a.dry_skin_score,
        "oily_skin_score": a.oily_skin_score,
        "dark_spots_score": a.dark_spots_score,
        "wrinkles_score": a.wrinkles_score,
        "redness_score": a.redness_score,
        "texture_score": a.texture_score,
        "hyperpigmentation_score": a.hyperpigmentation_score,
        "overall_score": a.overall_score,
        "ai_response": a.ai_response,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


def _scan_dict(scan: SkinScan, analysis: Optional[AnalysisResult] = None) -> dict:
    return {
        "id": scan.id,
        "user_id": scan.user_id,
        "image_path": scan.image_path,
        "status": scan.status,
        "image_quality_score": scan.image_quality_score,
        "created_at": scan.created_at.isoformat() if scan.created_at else None,
        "analysis": _analysis_dict(analysis),
    }


def _product_dict(p: Product) -> dict:
    # skin_concern / ingredients are stored as JSON strings or plain text
    # in the monolith; return the parsed list when it is valid JSON.
    def _maybe_json(value):
        if value is None:
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value

    return {
        "id": p.id,
        "brand": p.brand,
        "name": p.name,
        "category": p.category,
        "description": p.description,
        "ingredients": _maybe_json(p.ingredients),
        "skin_concern": _maybe_json(p.skin_concern),
        "usage_guidance": p.usage_guidance,
        "shopee_link": p.shopee_link,
        "tokopedia_link": p.tokopedia_link,
        "status": p.status,
    }


def _log_activity(db: Session, request: Request, user_id: int, label: str) -> None:
    """Best-effort activity log; never fails the request it logs."""
    try:
        log = ActivityLog(
            user_id=user_id,
            activity=label,
            ip_address=request.client.host if request.client else None,
        )
        db.add(log)
        db.commit()
    except Exception as e:  # noqa: BLE001 - logging must not break auth
        logger.warning("Failed to record activity log: %s", e)
        db.rollback()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LoginIn(BaseModel):
    email: EmailStr
    password: str


class RegisterIn(BaseModel):
    name: str
    email: EmailStr
    password: str
    skin_concerns: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Public health check, mirrors /api/v1/health. No DB access."""
    return JSONResponse(
        content={
            "status": "online",
            "app": settings.APP_NAME,
            "environment": settings.APP_ENV,
            "version": "2.0.0-api-only",
            "api": "v2",
            "features": {
                "templates": "removed - frontend is Next.js",
                "ai_integration": "lazy - model not required to import",
            },
        }
    )


@router.get("/products")
async def list_products(db: Session = Depends(get_db)):
    """Public: active products list (empty list is valid, not an error)."""
    try:
        products = (
            db.query(Product)
            .filter(Product.status == "active")
            .order_by(Product.id.asc())
            .all()
        )
    except Exception as e:  # noqa: BLE001 - DB down -> 503, not 500
        logger.warning("products query failed: %s", e)
        return _db_unavailable()
    return JSONResponse(content=[_product_dict(p) for p in products])


# ---------------------------------------------------------------------------
# Auth (session cookie, same shape as the monolith)
# ---------------------------------------------------------------------------

@router.post("/auth/login")
async def login(request: Request, payload: LoginIn, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    if not email or not payload.password:
        return JSONResponse(
            status_code=401, content={"detail": "Email dan password wajib diisi."}
        )
    try:
        user = db.query(User).filter(User.email == email).first()
    except Exception as e:  # noqa: BLE001
        logger.warning("login query failed: %s", e)
        return _db_unavailable()
    if not user or not verify_password(payload.password, user.password_hash):
        return JSONResponse(
            status_code=401, content={"detail": "Email atau password salah."}
        )
    if user.status != "active":
        return JSONResponse(
            status_code=401, content={"detail": "Akun dinonaktifkan."}
        )

    request.session["user"] = {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
    }
    _log_activity(
        db, request, user.id,
        "Admin Login" if user.role == "admin" else "Customer Login",
    )
    return JSONResponse(content={"user": _user_dict(user)})


@router.post("/auth/register", status_code=201)
async def register(request: Request, payload: RegisterIn, db: Session = Depends(get_db)):
    """Direct registration (OTP bypass, see module docstring).

    The User model has no ``phone`` / ``skin_concerns`` columns, so
    skin concerns are stored in ``profile_data`` JSON, exactly as the
    monolith's ``User.get_profile_data()`` expects.
    """
    name = payload.name.strip()
    email = payload.email.strip().lower()
    if not name or not email or not payload.password:
        return JSONResponse(
            status_code=400, content={"detail": "Nama, email, dan password wajib diisi."}
        )
    if len(payload.password) < 6:
        return JSONResponse(
            status_code=400, content={"detail": "Password minimal 6 karakter."}
        )
    try:
        if db.query(User).filter(User.email == email).first():
            return JSONResponse(
                status_code=409, content={"detail": "Email sudah terdaftar."}
            )
        profile = None
        if payload.skin_concerns:
            profile = json.dumps({"skin_concerns": payload.skin_concerns})
        user = User(
            name=name,
            email=email,
            password_hash=get_password_hash(payload.password),
            role="customer",
            status="active",
            profile_data=profile,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    except Exception as e:  # noqa: BLE001
        logger.warning("register failed: %s", e)
        db.rollback()
        return _db_unavailable()
    request.session["user"] = {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
    }
    return JSONResponse(status_code=201, content={"user": _user_dict(user)})


@router.post("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return JSONResponse(content={"status": "ok"})


@router.get("/me")
async def me(request: Request):
    """Current session user, or 401. No DB access, so this works offline."""
    user = _session_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"detail": "Login required"})
    return JSONResponse(content={"user": user})


# ---------------------------------------------------------------------------
# Customer: dashboard / history / scan
# ---------------------------------------------------------------------------

@router.get("/dashboard")
async def dashboard(request: Request, db: Session = Depends(get_db)):
    user = _require_session(request)
    if user.get("role") != "customer":
        raise _forbidden("Dashboard ini untuk customer.")
    try:
        scans = (
            db.query(SkinScan)
            .filter(SkinScan.user_id == user["id"])
            .order_by(SkinScan.created_at.desc())
            .limit(10)
            .all()
        )
        items = []
        for scan in scans:
            analysis = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.skin_scan_id == scan.id)
                .first()
            )
            items.append(_scan_dict(scan, analysis))
    except Exception as e:  # noqa: BLE001
        logger.warning("dashboard query failed: %s", e)
        return _db_unavailable()
    return JSONResponse(
        content={
            "user": user,
            "latest_scan": items[0] if items else None,
            "recent_scans": items,
        }
    )


@router.get("/history")
async def history(request: Request, limit: int = 50, db: Session = Depends(get_db)):
    user = _require_session(request)
    if user.get("role") != "customer":
        raise _forbidden("Riwayat ini untuk customer.")
    limit = max(1, min(limit, 100))
    try:
        scans = (
            db.query(SkinScan)
            .filter(SkinScan.user_id == user["id"])
            .order_by(SkinScan.created_at.desc())
            .limit(limit)
            .all()
        )
        items = []
        for scan in scans:
            analysis = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.skin_scan_id == scan.id)
                .first()
            )
            items.append(_scan_dict(scan, analysis))
    except Exception as e:  # noqa: BLE001
        logger.warning("history query failed: %s", e)
        return _db_unavailable()
    return JSONResponse(content={"scans": items, "total": len(items)})


@router.get("/history/{scan_id}")
async def history_detail(scan_id: int, request: Request, db: Session = Depends(get_db)):
    user = _require_session(request)
    if user.get("role") != "customer":
        raise _forbidden("Detail scan ini untuk customer.")
    try:
        scan = (
            db.query(SkinScan)
            .filter(SkinScan.id == scan_id, SkinScan.user_id == user["id"])
            .first()
        )
        if not scan:
            # 404 covers both "not found" and "belongs to another user",
            # so ownership is not leaked to the caller.
            return JSONResponse(status_code=404, content={"detail": "Scan not found"})
        analysis = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.skin_scan_id == scan.id)
            .first()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("history detail query failed: %s", e)
        return _db_unavailable()
    return JSONResponse(content=_scan_dict(scan, analysis))


@router.post("/scan/upload", status_code=201)
async def scan_upload(
    request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)
):
    """Minimal scan upload: validate + store + create a processing row.

    No AI analysis is run here and ``ai_service`` is never imported, so
    TensorFlow / the .h5 model are not required. A follow-up endpoint
    must pick up ``status="processing"`` rows and run the model.
    """
    user = _require_session(request)
    if user.get("role") != "customer":
        raise _forbidden("Upload scan ini untuk customer.")
    # Imported here (not at module level) so a missing Pillow install
    # fails only this endpoint, not the whole app import.
    from app.services.image_service import ImageService, ImageValidationException

    try:
        filename = await ImageService.process_and_save_scan(file)
    except ImageValidationException as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:  # noqa: BLE001
        logger.warning("image processing failed: %s", e)
        return JSONResponse(
            status_code=400, content={"detail": "Gagal memproses gambar."}
        )
    try:
        scan = SkinScan(
            user_id=user["id"],
            image_path=filename,
            status="processing",
            image_quality_score=0.9,
        )
        db.add(scan)
        db.commit()
        db.refresh(scan)
    except Exception as e:  # noqa: BLE001
        logger.warning("scan row creation failed: %s", e)
        db.rollback()
        return _db_unavailable()
    return JSONResponse(
        status_code=201,
        content={
            "scan_id": scan.id,
            "status": scan.status,
            "image_path": filename,
            "message": "Scan tersimpan, analisis AI belum dijalankan.",
        },
    )


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@router.get("/admin/summary")
async def admin_summary(request: Request, db: Session = Depends(get_db)):
    user = _require_session(request)
    if user.get("role") != "admin":
        raise _forbidden("Hanya admin.")
    try:
        total_users = db.query(User).count()
        total_scans = db.query(SkinScan).count()
        total_products = db.query(Product).count()
        active_products = (
            db.query(Product).filter(Product.status == "active").count()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("admin summary query failed: %s", e)
        return _db_unavailable()
    return JSONResponse(
        content={
            "users": total_users,
            "scans": total_scans,
            "products": total_products,
            "active_products": active_products,
        }
    )
