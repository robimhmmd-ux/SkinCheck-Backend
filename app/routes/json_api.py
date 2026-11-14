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
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.config import settings
from app.config.database import get_db
from app.models.activity_log import ActivityLog
from app.models.analysis_result import AnalysisResult
from app.models.product import Product
from app.models.skin_scan import SkinScan
from app.models.system_setting import SystemSetting
from app.models.user import User
from app.services.fonnte_otp import _send_whatsapp, mask_phone, normalize_phone
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


def _require_verified_customer(request: Request, db: Session, allow_guest: bool = False) -> dict:
    """Customer yang statusnya active (verifikasi OTP + biodata selesai).

    Guest (tanpa sesi) tetap lolos seperti sekarang kalau allow_guest —
    jalur demo Tester id=1 di endpoint upload. Customer yang login tapi
    belum verifikasi (pending / awaiting_biodata) ditolak 403, supaya
    kunci fitur di frontend ditegakkan juga server-side.
    """
    user = _session_user(request)
    if not user:
        if allow_guest:
            return {"id": 1, "name": "Tester", "role": "customer", "is_guest": True}
        raise _unauthorized()
    if user.get("role") != "customer":
        raise _forbidden("Fitur ini untuk customer.")
    row = db.query(User).filter(User.id == user["id"]).first()
    if not row or row.status != "active":
        raise _forbidden("Akun tidak diizinkan. Lakukan verifikasi kode OTP dulu, baru bisa memakai fitur ini.")
    return user


def _require_chat_user(request: Request, db: Session) -> dict:
    """Gate khusus chatbot: admin BOLEH chat (tanpa OTP), customer WAJIB active.

    Sebelumnya chat memakai _require_verified_customer, yang menolak admin
    (403 "Fitur ini untuk customer.") — regresi dari alur verifikasi
    2026-11-07: chatbot admin yang sudah jalan sejak 2026-11-05 jadi tidak
    bisa dipakai lewat widget. Admin tidak butuh verifikasi OTP; chatbot
    admin membaca data website (ADMIN_SYSTEM_PROMPT), bukan data pribadi.
    Customer yang belum verifikasi tetap ditolak 403 dengan pesan yang
    jelas, supaya widget menampilkan ajakan verifikasi OTP, bukan
    "Koneksi terputus".
    """
    user = _session_user(request)
    if not user:
        raise _unauthorized()
    if user.get("role") == "admin":
        return user
    if user.get("role") != "customer":
        raise _forbidden("Fitur ini untuk customer.")
    row = db.query(User).filter(User.id == user["id"]).first()
    if not row or row.status != "active":
        raise _forbidden("Akun tidak diizinkan. Lakukan verifikasi kode OTP dulu, baru bisa memakai fitur ini.")
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


def _iso_utc(dt: Optional[datetime]) -> Optional[str]:
    """Emit a timestamp as a UTC-aware ISO string.

    All models store naive datetimes from datetime.utcnow, i.e. UTC
    wall-clock without an offset. Emitting them bare made browsers parse
    them as LOCAL time, so WIB users saw history times 7 hours behind
    reality. Attaching +00:00 makes clients convert correctly.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


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
        "created_at": _iso_utc(a.created_at),
    }


def _scan_dict(scan: SkinScan, analysis: Optional[AnalysisResult] = None) -> dict:
    return {
        "id": scan.id,
        "user_id": scan.user_id,
        "image_path": scan.image_path,
        "status": scan.status,
        "image_quality_score": scan.image_quality_score,
        "created_at": _iso_utc(scan.created_at),
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


def _scan_gates(request: Request, db: Session, user: dict):
    """Return an error JSONResponse when a scan upload must be refused.

    Two admin-controlled gates (system_settings, read per request):
    - Maintenance mode (MAINTENANCE_MODE on/true/1): customer uploads
      are refused with 503. Admins are exempt from this block (they hit
      the customer-only role check below as before).
    - Daily quota (SCAN_DAILY_QUOTA positive int): logged-in customers
      are limited to that many scan rows created today (UTC day, matching
      the UTC storage). Unset/empty/non-positive = unlimited, so current
      customer behaviour is unchanged by default. Guest fallback (id=1)
      is exempt from quota — counting all guests together would lock the
      demo path for everyone after N scans.
    """
    try:
        maintenance = (SystemSetting.get_value(db, "MAINTENANCE_MODE", "off") or "").strip().lower()
        if maintenance in ("on", "true", "1") and user.get("role") != "admin":
            return JSONResponse(
                status_code=503,
                content={"detail": "Sistem sedang dalam mode pemeliharaan. Coba lagi nanti."},
            )
        if not user.get("is_guest"):
            raw_quota = (SystemSetting.get_value(db, "SCAN_DAILY_QUOTA", "") or "").strip()
            try:
                quota = int(raw_quota) if raw_quota else 0
            except ValueError:
                quota = 0
            if quota > 0:
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                used = (
                    db.query(SkinScan)
                    .filter(SkinScan.user_id == user["id"], SkinScan.created_at >= today_start)
                    .count()
                )
                if used >= quota:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": f"Kuota scan harian tercapai ({quota} scan/hari). Coba lagi besok."},
                    )
    except Exception as e:  # noqa: BLE001 - gate failure must not break uploads
        logger.warning("scan gates check failed (allowing upload): %s", e)
    return None


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
    if user.status not in ("active", "pending", "awaiting_biodata"):
        return JSONResponse(
            status_code=401, content={"detail": "Akun dinonaktifkan."}
        )
    # pending / awaiting_biodata BOLEH masuk: mereka mendarat di dashboard
    # dengan modal verifikasi (OTP dulu, lalu biodata) dan fitur terkunci
    # sampai statusnya active. Status ikut disimpan di sesi supaya /me tahu.

    request.session["user"] = {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "status": user.status,
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
        # Aturan nomor WA ganda (dikendalikan admin di Pengaturan, kunci
        # ALLOW_SAME_PHONE_NUMBER; default "off" = satu nomor hanya untuk
        # satu akun). Kalau admin mengaktifkannya ("on"), nomor yang sama
        # boleh dipakai beberapa akun — untuk demo presentasi.
        allow_same_phone = (
            SystemSetting.get_value(db, "ALLOW_SAME_PHONE_NUMBER", "off") or "off"
        ).strip().lower() == "on"
        if not allow_same_phone:
            for other in db.query(User).all():
                try:
                    other_data = json.loads(other.profile_data or "{}")
                except Exception:  # noqa: BLE001
                    other_data = {}
                if other_data.get("phone") == phone62:
                    return JSONResponse(
                        status_code=409,
                        content={"detail": "Nomor WhatsApp sudah digunakan akun lain."},
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
        # Jejak untuk notifikasi admin: akun baru daftar tapi belum verifikasi.
        _log_activity(db, request, user.id, "Mendaftar akun baru (belum verifikasi)")
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
async def logout(request: Request, db: Session = Depends(get_db)):
    # Hapus riwayat chatbot saat logout (roboy 2026-11-13): berlaku untuk
    # admin maupun customer, lewat jalur logout mana pun (sidebar, profil,
    # header) — penghapusan dijamin di server, bukan di pemanggil frontend.
    # Kegagalan hapus tidak boleh menggagalkan logout.
    user = _session_user(request)
    if user:
        try:
            from app.models.chat_message import ChatMessage

            db.query(ChatMessage).filter(ChatMessage.user_id == user["id"]).delete()
            db.commit()
        except Exception:
            db.rollback()
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
async def otp_verify(request: Request, payload: OtpVerifyIn, db: Session = Depends(get_db)):
    """Verifikasi kode OTP WhatsApp (6 digit, 10 menit, sekali pakai).

    Kalau kode benar, akun pending dengan nomor WA yang sama maju ke tahap
    biodata (pending -> awaiting_biodata) — BUKAN langsung active. Fitur
    penuh baru terbuka sesudah biodata dilengkapi (lihat
    POST /profile/biodata). Verifikasi mandiri tanpa akun (mis. nomor belum
    didaftarkan) tetap mengembalikan verified=True tanpa mengubah apa-apa.
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
    activated_ids = []
    try:
        for candidate in db.query(User).filter(User.status == "pending").all():
            try:
                data = json.loads(candidate.profile_data or "{}")
            except Exception:  # noqa: BLE001
                data = {}
            if data.get("phone") == phone62:
                # OTP lolos -> akun maju ke tahap biodata (BUKAN langsung
                # active). Fitur penuh baru terbuka sesudah biodata lengkap.
                candidate.status = "awaiting_biodata"
                activated_ids.append(candidate.id)
                activated = True
        if activated:
            db.commit()
            for uid in activated_ids:
                _log_activity(db, request, uid, "Verifikasi OTP WhatsApp berhasil")
            # Kalau pemverifikasi sedang login sebagai salah satu akun itu
            # (alur modal verifikasi di dashboard), segarkan status sesinya
            # supaya /me langsung tahu tahap berikutnya.
            sess = request.session.get("user")
            if isinstance(sess, dict) and sess.get("id") in activated_ids:
                sess["status"] = "awaiting_biodata"
                request.session["user"] = sess
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


class ProfileNameUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None  # diterima tapi tidak pernah diubah


@router.patch("/profile")
async def update_my_profile(
    request: Request, body: ProfileNameUpdate, db: Session = Depends(get_db)
):
    """Ubah nama akun sendiri. Email TIDAK dapat diubah lewat endpoint ini:
    permintaan yang membawa email berbeda dari email saat ini ditolak 400
    (bukan diabaikan diam-diam), supaya aturannya ditegakkan server-side
    dan tidak bisa diakali dari client."""
    user = _require_session(request)
    try:
        db_user = db.query(User).filter(User.id == user["id"]).first()
        if not db_user:
            return JSONResponse(status_code=404, content={"detail": "Akun tidak ditemukan."})
        if body.email is not None and body.email != db_user.email:
            return JSONResponse(
                status_code=400, content={"detail": "Email tidak dapat diubah."}
            )
        if body.name is not None:
            name = body.name.strip()
            if not name:
                return JSONResponse(
                    status_code=400, content={"detail": "Nama tidak boleh kosong."}
                )
            db_user.name = name
        db.commit()
        db.refresh(db_user)
    except Exception as e:  # noqa: BLE001
        logger.warning("profile update failed: %s", e)
        db.rollback()
        return _db_unavailable()
    # Segarkan sesi supaya getMe langsung menampilkan nama baru.
    sess = request.session.get("user")
    if isinstance(sess, dict):
        sess["name"] = db_user.name
        request.session["user"] = sess
    _log_activity(db, request, db_user.id, "Customer memperbarui profilnya")
    return JSONResponse(content={"user": _user_dict(db_user)})


class BiodataIn(BaseModel):
    name: str
    age: int
    gender: str


@router.post("/profile/biodata")
async def complete_biodata(request: Request, body: BiodataIn, db: Session = Depends(get_db)):
    """Lengkapi biodata akun (tahap verifikasi sesudah OTP).

    Hanya untuk akun berstatus awaiting_biodata (OTP sudah lolos) atau
    active (memperbarui biodata). Akun pending (OTP belum lolos) ditolak.
    Kalau biodata lengkap dan statusnya awaiting_biodata, status berubah
    menjadi active dan seluruh fitur customer terbuka.
    """
    user = _require_session(request)
    name = (body.name or "").strip()
    gender = (body.gender or "").strip()
    if not name:
        return JSONResponse(status_code=400, content={"detail": "Nama wajib diisi."})
    if gender not in ("Laki-laki", "Perempuan"):
        return JSONResponse(status_code=400, content={"detail": "Jenis kelamin harus 'Laki-laki' atau 'Perempuan'."})
    if not isinstance(body.age, int) or isinstance(body.age, bool) or not 5 <= body.age <= 120:
        return JSONResponse(status_code=400, content={"detail": "Umur harus di antara 5 dan 120 tahun."})
    try:
        db_user = db.query(User).filter(User.id == user["id"]).first()
        if not db_user:
            return JSONResponse(status_code=404, content={"detail": "Akun tidak ditemukan."})
        if db_user.status == "pending":
            return JSONResponse(status_code=400, content={"detail": "Verifikasi kode OTP WhatsApp dulu sebelum melengkapi biodata."})
        if db_user.status not in ("awaiting_biodata", "active"):
            return JSONResponse(status_code=400, content={"detail": "Akun dinonaktifkan."})
        try:
            profile_data = json.loads(db_user.profile_data or "{}")
        except Exception:  # noqa: BLE001
            profile_data = {}
        profile_data["age"] = body.age
        profile_data["gender"] = gender
        db_user.name = name
        db_user.profile_data = json.dumps(profile_data)
        completed = False
        if db_user.status == "awaiting_biodata":
            db_user.status = "active"
            completed = True
        db.commit()
        db.refresh(db_user)
    except Exception as e:  # noqa: BLE001
        logger.warning("biodata update failed: %s", e)
        db.rollback()
        return _db_unavailable()
    # Segarkan sesi supaya getMe langsung menampilkan nama + status baru.
    sess = request.session.get("user")
    if isinstance(sess, dict):
        sess["name"] = db_user.name
        sess["status"] = db_user.status
        request.session["user"] = sess
    if completed:
        _log_activity(db, request, db_user.id, "Melengkapi biodata — verifikasi akun selesai")
    return JSONResponse(content={"user": _user_dict(db_user), "verification_complete": completed})


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


def _notify_scan_ready_wa(db: Session, user: dict, analysis_result: AnalysisResult) -> None:
    """Kirim WA 'hasil scan siap' — best-effort, tidak pernah gagalkan scan.

    Hanya untuk customer login yang punya nomor HP di profile_data["phone"]
    (format 62..., dinormalisasi saat register). Tamu/tester (is_guest)
    dilewati. Kegagalan kirim apa pun (token Fonnte belum disetel, Fonnte
    down, nomor tidak valid) hanya dicatat sebagai warning; respons scan
    tetap sukses.
    """
    try:
        if user.get("is_guest"):
            return
        if analysis_result is None or analysis_result.overall_score is None:
            return
        row = db.query(User).filter(User.id == user["id"]).first()
        if row is None:
            return
        phone_raw = (row.get_profile_data() or {}).get("phone")
        if not phone_raw:
            return
        phone62 = normalize_phone(str(phone_raw))
        name = (row.name or user.get("name") or "").strip() or "Kak"
        overall = float(analysis_result.overall_score)
        message = (
            f"Halo {name}! Hasil scan wajah kamu sudah siap — "
            f"skor kesehatan kulitmu {overall:g}/100. "
            "Buka Laomera untuk melihat detailnya."
        )
        _send_whatsapp(phone62, message, db)
        logger.info(
            "Notifikasi WA 'hasil scan siap' terkirim ke %s",
            mask_phone(phone62),
        )
    except Exception as e:  # noqa: BLE001 - notifikasi tidak boleh gagalkan scan
        logger.warning("Notifikasi WA 'hasil scan siap' gagal: %s", e)
        return


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
    user = _require_verified_customer(request, db, allow_guest=True)
    # Admin-controlled gates (maintenance mode / daily quota).
    gate_error = _scan_gates(request, db, user)
    if gate_error is not None:
        return gate_error
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

    if scan.status == "completed" and analysis_result is not None:
        _notify_scan_ready_wa(db, user, analysis_result)

    response_content = {
        "scan_id": scan.id,
        "status": scan.status,
        "image_path": filename,
        "analysis": _analysis_dict(analysis_result),
        "is_guest": user.get("is_guest", False),
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
    request: Request,
    files: List[UploadFile] = File(...),
    poses: List[str] = Form([]),
    db: Session = Depends(get_db),
):
    """Synchronous multi-frame upload and multi-pose AI skin analysis.

    Receives multiple webcam frames (e.g. 10 fps over multi-pose capture),
    filters frames with YuNet face detection and pose estimation, samples
    best frames per pose, executes batch VGG16 skin condition inference,
    and stores 1 SkinScan + 1 AnalysisResult row.
    """
    user = _require_verified_customer(request, db, allow_guest=True)
    # Admin-controlled gates (maintenance mode / daily quota).
    gate_error = _scan_gates(request, db, user)
    if gate_error is not None:
        return gate_error
    if not files:
        return JSONResponse(
            status_code=400, content={"detail": "Minimal 1 frame foto harus diunggah."}
        )

    from app.services.image_service import ImageService

    # Label pose sejajar dengan urutan file; kalau jumlahnya tidak cocok,
    # label diabaikan (provider mengelompokkan semua frame jadi satu grup).
    # Frame yang gagal validasi dilewati bersama labelnya supaya urutan
    # frame dan label tidak bergeser.
    poses_aligned = poses if len(poses) == len(files) else [None] * len(files)
    saved_filenames = []
    saved_abs_paths = []
    saved_poses = []
    for f, pose in zip(files, poses_aligned):
        try:
            fname = await ImageService.process_and_save_scan(f)
            saved_filenames.append(fname)
            saved_abs_paths.append(str(ImageService.get_image_path(fname)))
            saved_poses.append(pose)
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
        ai_data = provider.analyze_multi_frame_skin_condition(saved_abs_paths, saved_poses)

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

    if scan.status == "completed" and analysis_result is not None:
        _notify_scan_ready_wa(db, user, analysis_result)

    response_content = {
        "scan_id": scan.id,
        "status": scan.status,
        "image_path": scan.image_path,
        "analysis": _analysis_dict(analysis_result),
        "frames_captured": len(files),
        "frames_passed_yunet": ai_data.get("frames_passed_yunet", 0),
        "frames_used_vgg": ai_data.get("frames_used_vgg", 0),
        "per_pose": ai_data.get("per_pose", {}),
        "statements": ai_data.get("statements", []),
        "is_guest": user.get("is_guest", False),
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
    user = _require_chat_user(request, db)
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


@router.post("/assistant/chat/stream")
def assistant_chat_stream(request: Request, payload: AssistantChatIn, db: Session = Depends(get_db)):
    """Streaming chat (SSE): the first tokens reach the browser as soon as
    Gemini starts generating, instead of waiting for the full reply.

    Sync route (not async): the generator does blocking Gemini/DB work, and
    FastAPI runs sync routes + sync iterators in its threadpool, so the DB
    session stays usable for the whole stream. The session dependency is
    closed only after the stream finishes, which is also when history is
    saved (inside chat_stream).
    """
    user = _require_chat_user(request, db)
    message = payload.message.strip()
    if not message:
        return JSONResponse(status_code=400, content={"detail": "Pesan tidak boleh kosong."})
    if len(message) > 2000:
        return JSONResponse(status_code=400, content={"detail": "Pesan terlalu panjang (maksimal 2000 karakter)."})

    from app.services.ai_assistant_service import assistant

    def event_gen():
        try:
            for delta in assistant.chat_stream(
                user_id=user["id"],
                message=message,
                db=db,
                scan_id=payload.scan_id,
                role=user.get("role", "customer"),
            ):
                if delta:
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            yield "data: {\"done\": true}\n\n"
        except Exception as e:
            logger.error("AI assistant stream error: %s", e)
            yield f"data: {json.dumps({'error': 'Terjadi kesalahan pada layanan AI.'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
                "created_at": _iso_utc(m.created_at),
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
