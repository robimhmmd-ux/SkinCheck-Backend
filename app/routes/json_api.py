"""SkinCheck API v2 — JSON-only API for the Next.js frontend.

This is the API-only layer of the split backend. It deliberately never
returns a TemplateResponse: every endpoint returns JSONResponse, so the
Next.js frontend can consume it without any Jinja template.

Auth model (kept identical to the monolith, see AuthController):
- Session cookie ``skincheck_session`` via Starlette SessionMiddleware.
- ``request.session["user"]`` is ``{"id", "name", "email", "role"}``.
- Passwords are bcrypt-hashed via ``app.utils.auth`` (same as monolith).

OTP note:
- Registration is OTP-gated: ``POST /api/v2/auth/register`` creates the
  account with status ``pending`` and NO session. WhatsApp OTP
  (``POST /api/v2/auth/otp/request`` + ``POST /api/v2/auth/otp/verify``,
  Fonnte, see ``app/services/fonnte_otp.py``) activates it
  (pending -> active); ``/auth/login`` rejects pending accounts until
  then. The OTP code is never returned in an API response.

AI / scan note:
- ``POST /api/v2/scan/upload`` runs synchronous AI analysis when local
  dependencies (TensorFlow, OpenCV) and the .h5 model are present. On Vercel
  or environments missing the model/dependencies, it gracefully marks the
  scan as failed without crashing the server.

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
    # Nomor WhatsApp untuk OTP Fonnte — WAJIB sejak akun dibuat pending
    # sampai verifikasi OTP. User tidak punya kolom phone, jadi nomor
    # yang sudah dinormalisasi (62...) disimpan di profile_data JSON,
    # sama seperti skin_concerns.
    phone: str


class OtpRequestIn(BaseModel):
    phone: str
    name: Optional[str] = None


class OtpVerifyIn(BaseModel):
    phone: str
    code: str


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Public health check, mirrors /api/v1/health. No DB access."""
    ai_status = "unavailable in this environment (Vercel / missing dependencies)"
    try:
        from pathlib import Path
        model_path = Path(__file__).resolve().parents[2] / "facial-condition-model" / "finetuned_best.h5"
        if model_path.exists():
            import tensorflow  # noqa: F401
            import cv2  # noqa: F401
            ai_status = "local synchronous Keras VGG16 model active"
    except Exception:
        ai_status = "unavailable in this environment (Vercel / missing dependencies)"

    return JSONResponse(
        content={
            "status": "online",
            "app": settings.APP_NAME,
            "environment": settings.APP_ENV,
            "version": "2.0.0-api-only",
            "api": "v2",
            "features": {
                "templates": "removed - frontend is Next.js",
                "ai_integration": ai_status,
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
    if user.status == "pending":
        return JSONResponse(
            status_code=401,
            content={"detail": "Akun belum terverifikasi. Verifikasi kode OTP WhatsApp dulu di halaman verifikasi, lalu coba masuk lagi."},
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
    """Registration yang digerbang OTP WhatsApp.

    Akun dibuat dengan status ``pending`` dan TIDAK diberi sesi login
    sampai kode OTP WhatsApp terverifikasi (lihat /auth/otp/verify,
    yang mengubah pending -> active). Jadi "berhasil daftar" tidak lagi
    berarti "bisa langsung masuk". The User model has no ``phone`` /
    ``skin_concerns`` columns, so skin concerns and the WhatsApp number
    are stored in ``profile_data`` JSON, exactly as the monolith's
    ``User.get_profile_data()`` expects. Nomor WA inilah yang dipakai
    halaman /verify-otp untuk OTP Fonnte.
    """
    name = payload.name.strip()
    email = payload.email.strip().lower()
    if not name or not email or not payload.password:
        return JSONResponse(
            status_code=400, content={"detail": "Nama, email, dan password wajib diisi."}
        )
    if not payload.phone or not payload.phone.strip():
        return JSONResponse(
            status_code=400, content={"detail": "Nomor WhatsApp wajib diisi untuk verifikasi OTP."}
        )
    if len(payload.password) < 6:
        return JSONResponse(
            status_code=400, content={"detail": "Password minimal 6 karakter."}
        )
    try:
        from app.services.fonnte_otp import mask_phone, normalize_phone

        phone62 = normalize_phone(payload.phone)
    except Exception as e:  # noqa: BLE001 - InvalidPhoneError
        return JSONResponse(status_code=400, content={"detail": str(e)})
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            if existing.status == "pending":
                return JSONResponse(
                    status_code=409,
                    content={"detail": "Email sudah terdaftar tapi belum terverifikasi. Buka halaman verifikasi OTP WhatsApp dan verifikasi nomor kamu."},
                )
            return JSONResponse(
                status_code=409, content={"detail": "Email sudah terdaftar."}
            )
        profile_data = {}
        if payload.skin_concerns:
            profile_data["skin_concerns"] = payload.skin_concerns
        profile_data["phone"] = phone62
        user = User(
            name=name,
            email=email,
            password_hash=get_password_hash(payload.password),
            role="customer",
            status="pending",
            profile_data=json.dumps(profile_data),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    except Exception as e:  # noqa: BLE001
        logger.warning("register failed: %s", e)
        db.rollback()
        return _db_unavailable()
    # Sengaja TIDAK membuat sesi: akun pending tidak boleh masuk
    # sebelum OTP terverifikasi.
    return JSONResponse(
        status_code=201,
        content={
            "user": _user_dict(user),
            "otp_required": True,
            "phone_masked": mask_phone(phone62),
        },
    )


@router.post("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return JSONResponse(content={"status": "ok"})


@router.post("/auth/otp/request")
async def otp_request(payload: OtpRequestIn, db: Session = Depends(get_db)):
    """Kirim kode OTP 6 digit via WhatsApp (Fonnte) ke nomor yang diminta.

    Tidak pernah mengembalikan kode di respons. Kalau token Fonnte
    belum dikonfigurasi, balas 503 yang jelas (bukan klaim "terkirim").
    Register /api/v2 tetap langsung tanpa gate OTP — endpoint ini
    verifikasi WhatsApp mandiri, dipakai halaman /verify-otp.
    """
    from app.services.fonnte_otp import (
        FonnteNotConfiguredError,
        FonnteSendError,
        InvalidPhoneError,
        TooSoonError,
        mask_phone,
        normalize_phone,
        request_whatsapp_otp,
    )

    try:
        result = request_whatsapp_otp(db, payload.phone, payload.name or "")
    except InvalidPhoneError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except TooSoonError as e:
        return JSONResponse(status_code=429, content={"detail": str(e)})
    except FonnteNotConfiguredError:
        return JSONResponse(
            status_code=503,
            content={"detail": "Token Fonnte belum dikonfigurasi. Isi FONNTE_TOKEN di Environment Variables backend (atau WA_API_TOKEN di pengaturan sistem), lalu redeploy backend."},
        )
    except FonnteSendError as e:
        return JSONResponse(
            status_code=503,
            content={"detail": str(e) or "Gagal mengirim WhatsApp via Fonnte. Periksa token Fonnte dan nomor perangkat Fonnte, lalu coba lagi."},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("otp request failed: %s", e)
        db.rollback()
        return _db_unavailable()
    return JSONResponse(content={"status": "sent", **result})


@router.post("/auth/otp/verify")
async def otp_verify(payload: OtpVerifyIn, db: Session = Depends(get_db)):
    """Verifikasi kode OTP WhatsApp (6 digit, 10 menit, sekali pakai).

    Kalau kode benar, akun pending dengan nomor WA yang sama diaktifkan
    (pending -> active) sehingga baru bisa login. Verifikasi mandiri
    tanpa akun (mis. nomor belum didaftarkan) tetap mengembalikan
    verified=True tanpa mengaktifkan apa-apa.
    """
    from app.services.fonnte_otp import (
        InvalidPhoneError,
        mask_phone,
        normalize_phone,
        verify_whatsapp_otp,
    )

    try:
        ok = verify_whatsapp_otp(db, payload.phone, payload.code)
        phone62 = normalize_phone(payload.phone)
        phone_masked = mask_phone(phone62)
    except InvalidPhoneError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:  # noqa: BLE001
        logger.warning("otp verify failed: %s", e)
        db.rollback()
        return _db_unavailable()
    if not ok:
        return JSONResponse(
            status_code=400,
            content={"detail": "Kode OTP salah, sudah dipakai, atau sudah kedaluwarsa. Minta kode baru."},
        )
    activated = False
    try:
        for candidate in db.query(User).filter(User.status == "pending").all():
            try:
                data = json.loads(candidate.profile_data or "{}")
            except Exception:  # noqa: BLE001
                data = {}
            if data.get("phone") == phone62:
                candidate.status = "active"
                activated = True
        if activated:
            db.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning("otp activation failed: %s", e)
        db.rollback()
    return JSONResponse(
        content={"verified": True, "phone_masked": phone_masked, "account_activated": activated}
    )


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
    """Synchronous scan upload and AI skin analysis.

    Validates and stores the image, creates a SkinScan row, and immediately
    executes KerasComputerVisionProvider if local AI dependencies and the .h5
    model are available. If AI inference fails or is not available, status
    is updated to 'failed' and analysis remains null without crashing the server.
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

    # Synchronous AI analysis (Option 1: local execution only)
    analysis_result = None
    try:
        from app.services.ai_service import KerasComputerVisionProvider

        abs_image_path = str(ImageService.get_image_path(filename))
        provider = KerasComputerVisionProvider()
        ai_data = provider.analyze_skin_condition(abs_image_path)

        analysis_result = AnalysisResult(
            skin_scan_id=scan.id,
            acne_score=float(ai_data.get("acne_score", 0.0)),
            dry_skin_score=float(ai_data.get("dry_skin_score", 0.0)),
            oily_skin_score=float(ai_data.get("oily_skin_score", 0.0)),
            dark_spots_score=float(ai_data.get("dark_spots_score", 0.0)),
            wrinkles_score=float(ai_data.get("wrinkles_score", 0.0)),
            redness_score=float(ai_data.get("redness_score", 0.0)),
            texture_score=float(ai_data.get("texture_score", 0.0)),
            hyperpigmentation_score=float(ai_data.get("hyperpigmentation_score", 0.0)),
            overall_score=float(ai_data.get("overall_score", 0.0)),
            ai_response=json.dumps(
                {
                    "acne_score": ai_data.get("acne_score"),
                    "dry_skin_score": ai_data.get("dry_skin_score"),
                    "oily_skin_score": ai_data.get("oily_skin_score"),
                    "dark_spots_score": ai_data.get("dark_spots_score"),
                    "wrinkles_score": ai_data.get("wrinkles_score"),
                    "overall_score": ai_data.get("overall_score"),
                    "confidence": ai_data.get("confidence"),
                    "frames_analyzed": ai_data.get("frames_analyzed"),
                    "analysis_status": ai_data.get("analysis_status"),
                }
            ),
        )
        db.add(analysis_result)
        scan.status = "completed"
        db.commit()
        db.refresh(scan)
        db.refresh(analysis_result)
    except Exception as e:  # noqa: BLE001
        logger.warning("AI skin analysis failed for scan %s: %s", scan.id, e)
        error_msg = str(e)
        db.rollback()
        try:
            db.query(SkinScan).filter(SkinScan.id == scan.id).update({"status": "failed"})
            db.commit()
            db.refresh(scan)
        except Exception as update_err:  # noqa: BLE001
            logger.warning("Failed to mark scan as failed: %s", update_err)
            db.rollback()
            scan.status = "failed"
        analysis_result = None

    response_content = {
        "scan_id": scan.id,
        "status": scan.status,
        "image_path": filename,
        "analysis": _analysis_dict(analysis_result),
    }
    if scan.status == "completed":
        response_content["message"] = "Scan berhasil dianalisis."
    else:
        response_content["message"] = error_msg or "Analisis AI gagal diproses."

    return JSONResponse(
        status_code=201,
        content=response_content,
    )


@router.post("/scan/multi-upload", status_code=201)
async def scan_multi_upload(
    request: Request, files: List[UploadFile] = File(...), db: Session = Depends(get_db)
):
    """Synchronous multi-frame upload and multi-pose AI skin analysis.

    Receives multiple webcam frames (e.g. 10 fps over multi-pose capture),
    filters frames with YuNet face detection and pose estimation, samples
    best frames per pose, executes batch VGG16 skin condition inference,
    and stores 1 SkinScan + 1 AnalysisResult row.
    """
    user = _require_session(request)
    if user.get("role") != "customer":
        raise _forbidden("Upload scan ini untuk customer.")
    if not files:
        return JSONResponse(
            status_code=400, content={"detail": "Minimal 1 frame foto harus diunggah."}
        )

    from app.services.image_service import ImageService

    saved_filenames = []
    saved_abs_paths = []
    for f in files:
        try:
            fname = await ImageService.process_and_save_scan(f)
            saved_filenames.append(fname)
            saved_abs_paths.append(str(ImageService.get_image_path(fname)))
        except Exception as e:  # noqa: BLE001
            logger.debug("Frame validation skipped: %s", e)
            continue

    if not saved_filenames:
        return JSONResponse(
            status_code=400,
            content={"detail": "Tidak ada frame gambar valid yang dapat diproses."},
        )

    # Create scan row with first representative frame
    try:
        scan = SkinScan(
            user_id=user["id"],
            image_path=saved_filenames[0],
            status="processing",
            image_quality_score=0.9,
        )
        db.add(scan)
        db.commit()
        db.refresh(scan)
    except Exception as e:  # noqa: BLE001
        logger.warning("multi-scan row creation failed: %s", e)
        db.rollback()
        return _db_unavailable()

    analysis_result = None
    ai_data = {}
    error_msg = None
    try:
        from app.services.ai_service import KerasComputerVisionProvider

        provider = KerasComputerVisionProvider()
        ai_data = provider.analyze_multi_frame_skin_condition(saved_abs_paths)

        analysis_result = AnalysisResult(
            skin_scan_id=scan.id,
            acne_score=float(ai_data.get("acne_score", 0.0)),
            dry_skin_score=float(ai_data.get("dry_skin_score", 0.0)),
            oily_skin_score=float(ai_data.get("oily_skin_score", 0.0)),
            dark_spots_score=float(ai_data.get("dark_spots_score", 0.0)),
            wrinkles_score=float(ai_data.get("wrinkles_score", 0.0)),
            redness_score=float(ai_data.get("redness_score", 0.0)),
            texture_score=float(ai_data.get("texture_score", 0.0)),
            hyperpigmentation_score=float(ai_data.get("hyperpigmentation_score", 0.0)),
            overall_score=float(ai_data.get("overall_score", 0.0)),
            ai_response=json.dumps(ai_data),
        )
        db.add(analysis_result)
        scan.status = "completed"
        db.commit()
        db.refresh(scan)
        db.refresh(analysis_result)
    except Exception as e:  # noqa: BLE001
        logger.warning("AI multi-frame skin analysis failed for scan %s: %s", scan.id, e)
        error_msg = str(e)
        db.rollback()
        try:
            db.query(SkinScan).filter(SkinScan.id == scan.id).update({"status": "failed"})
            db.commit()
            db.refresh(scan)
        except Exception as update_err:  # noqa: BLE001
            logger.warning("Failed to mark multi-scan as failed: %s", update_err)
            db.rollback()
            scan.status = "failed"
        analysis_result = None

    response_content = {
        "scan_id": scan.id,
        "status": scan.status,
        "image_path": scan.image_path,
        "analysis": _analysis_dict(analysis_result),
        "frames_captured": len(files),
        "frames_passed_yunet": ai_data.get("frames_passed_yunet", 0),
        "frames_used_vgg": ai_data.get("frames_used_vgg", 0),
    }
    if scan.status == "completed":
        response_content["message"] = "Scan multi-pose berhasil dianalisis."
    else:
        response_content["message"] = error_msg or "Analisis AI multi-frame gagal diproses."

    return JSONResponse(status_code=201, content=response_content)


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


# ---------------------------------------------------------------------------
# AI Assistant (Chatbot)
# ---------------------------------------------------------------------------

class AssistantChatIn(BaseModel):
    message: str
    scan_id: Optional[int] = None


@router.post("/assistant/chat")
async def assistant_chat(
    request: Request, payload: AssistantChatIn, db: Session = Depends(get_db)
):
    """API endpoint for AI skincare assistant chat."""
    user = _require_session(request)
    message = payload.message.strip()
    if not message:
        return JSONResponse(status_code=400, content={"detail": "Pesan tidak boleh kosong."})
    if len(message) > 2000:
        return JSONResponse(status_code=400, content={"detail": "Pesan terlalu panjang (maksimal 2000 karakter)."})

    try:
        from app.services.ai_assistant_service import assistant

        response_text = assistant.chat(
            user_id=user["id"],
            message=message,
            db=db,
            scan_id=payload.scan_id,
            role=user.get("role", "customer"),
        )
        return JSONResponse(status_code=200, content={"status": "success", "response": response_text})
    except Exception as e:
        logger.error("AI assistant chat error: %s", e)
        return JSONResponse(status_code=500, content={"detail": "Terjadi kesalahan pada layanan AI."})


@router.get("/assistant/history")
async def assistant_history(
    request: Request, db: Session = Depends(get_db)
):
    """Retrieve chat history for the authenticated user."""
    user = _session_user(request)
    if not user:
        return JSONResponse(status_code=200, content={"status": "success", "history": [], "authenticated": False})

    try:
        from app.models.chat_message import ChatMessage

        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == user["id"])
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        history = [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "time": m.created_at.strftime("%H:%M") if m.created_at else "",
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ]
        return JSONResponse(status_code=200, content={"status": "success", "history": history, "authenticated": True})
    except Exception as e:
        logger.warning("assistant history query failed: %s", e)
        return _db_unavailable()


@router.post("/assistant/clear")
async def assistant_clear(
    request: Request, db: Session = Depends(get_db)
):
    """Clear chat conversation history for the authenticated user."""
    user = _require_session(request)
    try:
        from app.models.chat_message import ChatMessage

        db.query(ChatMessage).filter(ChatMessage.user_id == user["id"]).delete()
        db.commit()
        return JSONResponse(status_code=200, content={"status": "success", "message": "Riwayat percakapan berhasil dibersihkan."})
    except Exception as e:
        logger.warning("clear chat history failed: %s", e)
        db.rollback()
        return _db_unavailable()
