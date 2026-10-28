from fastapi import Request, Form, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.config import settings
from app.config.database import get_db
from app.models.user import User
from app.models.activity_log import ActivityLog
from app.utils.auth import verify_password, get_password_hash
from app.services.otp_service import send_registration_otp, verify_otp
import json
import logging

templates = Jinja2Templates(directory=str(settings.TEMPLATE_DIR))
logger = logging.getLogger(__name__)


class AuthController:
    """
    Controller handling authentication for Customer and Admin.
    Customer registration now uses a two-step OTP email verification flow:
      Step 1: Fill form  →  data stored temporarily in session  →  OTP sent via Gmail SMTP
      Step 2: Enter OTP  →  account created  →  redirect to login
    """

    # ──────────────────────────────────────────────────────
    # CUSTOMER LOGIN
    # ──────────────────────────────────────────────────────

    @staticmethod
    async def show_customer_login(request: Request, error: str = None, success: str = None):
        """Render Unified Login page."""
        return templates.TemplateResponse("pages/auth/login.html", {
            "request": request,
            "page_title": "Masuk ke Akun — SkinCheck",
            "app_name": settings.APP_NAME,
            "error": error,
            "success": success
        })

    @staticmethod
    async def handle_customer_login(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_db)
    ):
        """Process Unified Login for both Customer and Admin."""
        email_clean = email.strip().lower() if email else ""
        if not email_clean or not password:
            return await AuthController.show_customer_login(request, error="Email dan password wajib diisi.")

        user = db.query(User).filter(User.email == email_clean).first()
        if not user:
            return await AuthController.show_customer_login(request, error="Email belum terdaftar. Silakan buat akun baru terlebih dahulu.")

        if not verify_password(password, user.password_hash):
            return await AuthController.show_customer_login(request, error="Password yang dimasukkan salah.")

        if user.status != "active":
            return await AuthController.show_customer_login(request, error="Akun Anda dinonaktifkan.")

        # Set session
        request.session["user"] = {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role
        }

        # Log activity safely (non-blocking)
        activity_label = "Admin Login" if user.role == "admin" else "Customer Login"
        try:
            log = ActivityLog(user_id=user.id, activity=activity_label, ip_address=request.client.host if request.client else None)
            db.add(log)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to record login activity log: {e}")
            db.rollback()

        # Automatic redirection based on account role
        if user.role == "admin":
            return RedirectResponse(url="/admin/dashboard", status_code=303)
        return RedirectResponse(url="/dashboard", status_code=303)

    # ──────────────────────────────────────────────────────
    # CUSTOMER REGISTRATION — STEP 1: Fill Form & Send OTP
    # ──────────────────────────────────────────────────────

    @staticmethod
    async def show_customer_register(request: Request, error: str = None):
        """Render Customer Registration page."""
        return templates.TemplateResponse("pages/auth/register.html", {
            "request": request,
            "page_title": "Daftar Akun Baru — SkinCheck",
            "app_name": settings.APP_NAME,
            "error": error
        })

    @staticmethod
    async def handle_customer_register(
        request: Request,
        name: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
        channel: str = Form("email"),
        phone: str = Form(None),
        skin_concerns: list[str] = Form([]),
        consent_privacy: bool = Form(False),
        consent_camera: bool = Form(False),
        db: Session = Depends(get_db)
    ):
        """
        STEP 1: Validate registration data, store in session, send OTP (Email/WhatsApp), redirect to OTP page.
        """
        if not consent_privacy:
            return await AuthController.show_customer_register(
                request, error="Anda harus menyetujui Kebijakan Privasi & Ketentuan Non-Medis.")

        if len(password) < 6:
            return await AuthController.show_customer_register(
                request, error="Password minimal 6 karakter.")

        email_clean = email.strip().lower()
        name_clean  = name.strip()

        if db.query(User).filter(User.email == email_clean).first():
            return await AuthController.show_customer_register(
                request, error="Email sudah terdaftar. Silakan gunakan email lain atau masuk.")

        from app.models.system_setting import SystemSetting
        otp_mode = SystemSetting.get_value(db, "REGISTRATION_OTP_MODE", "dual")
        if otp_mode == "bypass":
            # Flow Mode Bypass: Langsung aktif tanpa verifikasi OTP (Fitur kontrol Admin)
            new_user = User(
                name=name_clean,
                email=email_clean,
                phone=phone,
                password_hash=get_password_hash(password),
                skin_concerns=",".join(skin_concerns) if skin_concerns else "",
                role="customer",
                status="active"
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            request.session["user"] = {
                "id": new_user.id,
                "name": new_user.name,
                "email": new_user.email,
                "role": new_user.role
            }
            return RedirectResponse(url="/dashboard?welcome=1", status_code=303)

        effective_channel = channel
        if otp_mode == "email_only":
            effective_channel = "email"
        elif otp_mode == "wa_only":
            effective_channel = "whatsapp"

        # Temporarily store registration data in session (NOT committed to DB yet)
        request.session["pending_register"] = {
            "name": name_clean,
            "email": email_clean,
            "phone": phone,
            "channel": effective_channel,
            "password_hash": get_password_hash(password),
            "skin_concerns": skin_concerns,
            "consent_camera": consent_camera
        }

        # Send OTP via selected channel (Email or WhatsApp)
        result = send_registration_otp(email_clean, name_clean, db, channel=effective_channel, phone=phone)

        # Redirect to OTP verification page
        dev_hint = f"&dev_code={result['dev_code']}" if result.get("dev_code") else ""
        return RedirectResponse(
            url=f"/auth/verify-otp?email={email_clean}&channel={effective_channel}{dev_hint}",
            status_code=303
        )

    # ──────────────────────────────────────────────────────
    # CUSTOMER REGISTRATION — STEP 2: OTP Verification Page
    # ──────────────────────────────────────────────────────

    @staticmethod
    async def show_verify_otp(
        request: Request,
        email: str = None,
        channel: str = "email",
        dev_code: str = None,
        error: str = None,
        success: str = None
    ):
        """Render OTP verification page with Email & WhatsApp support."""
        return templates.TemplateResponse("pages/auth/verify_otp.html", {
            "request": request,
            "page_title": "Verifikasi OTP — SkinCheck",
            "app_name": settings.APP_NAME,
            "email": email or "",
            "channel": channel or "email",
            "dev_code": dev_code or "",
            "error": error,
            "success": success
        })

    @staticmethod
    async def handle_verify_otp(
        request: Request,
        email: str = Form(...),
        otp_code: str = Form(...),
        db: Session = Depends(get_db)
    ):
        """
        STEP 2: Validate OTP and create the user account.
        """
        pending = request.session.get("pending_register")
        email_clean = email.strip().lower()

        # Guard: session must match
        if not pending or pending.get("email") != email_clean:
            return RedirectResponse(url="/register?error=Sesi pendaftaran tidak ditemukan. Silakan daftar ulang.", status_code=303)

        code_clean = otp_code.strip()

        if not verify_otp(email_clean, code_clean, "register", db):
            return await AuthController.show_verify_otp(
                request, email=email_clean,
                error="Kode OTP salah atau sudah kedaluwarsa. Periksa kembali email Anda."
            )

        # OTP valid → create user
        new_user = User(
            name=pending["name"],
            email=email_clean,
            password_hash=pending["password_hash"],
            role="customer",
            status="active"
        )
        new_user.set_profile_data({
            "skin_concerns": pending.get("skin_concerns", []),
            "consent_camera": pending.get("consent_camera", False)
        })

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # Log
        log = ActivityLog(
            user_id=new_user.id,
            activity="Customer Registration (OTP Verified)",
            ip_address=request.client.host if request.client else None
        )
        db.add(log)
        db.commit()

        # Clear pending session
        request.session.pop("pending_register", None)

        return RedirectResponse(
            url="/login?success=Akun berhasil dibuat! Selamat datang di SkinCheck. Silakan masuk.",
            status_code=303
        )

    # ──────────────────────────────────────────────────────
    # RESEND OTP (AJAX endpoint)
    # ──────────────────────────────────────────────────────

    @staticmethod
    async def resend_otp(
        request: Request,
        db: Session = Depends(get_db)
    ):
        """API endpoint: resend OTP for pending registration."""
        try:
            body = await request.json()
            email_clean = (body.get("email") or "").strip().lower()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Request tidak valid."})

        pending = request.session.get("pending_register")
        if not pending or pending.get("email") != email_clean:
            return JSONResponse(status_code=400, content={"error": "Sesi pendaftaran tidak ditemukan."})

        channel = body.get("channel", pending.get("channel", "email"))
        phone = body.get("phone", pending.get("phone", None))
        result = send_registration_otp(email_clean, pending["name"], db, channel=channel, phone=phone)

        channel_label = "WhatsApp" if channel == "whatsapp" else "Email"
        if result.get("dev_code"):
            return {"status": "ok", "channel": channel, "message": f"OTP baru ({channel_label}) digenerate (mode dev).", "dev_code": result["dev_code"]}

        return {"status": "ok", "channel": channel, "message": f"OTP baru telah dikirim via {channel_label}."}

    # ──────────────────────────────────────────────────────
    # ADMIN LOGIN
    # ──────────────────────────────────────────────────────

    @staticmethod
    async def show_admin_login(request: Request, error: str = None):
        """Render dedicated Admin Login Console as specified in PRD v2.0 (FR-A04)."""
        user = request.session.get("user")
        if user and user.get("role") == "admin":
            return RedirectResponse(url="/admin/dashboard", status_code=303)
        return templates.TemplateResponse("pages/admin/login.html", {
            "request": request,
            "page_title": "Masuk ke Admin Console — SkinCheck",
            "error": error
        })

    @staticmethod
    async def handle_admin_login(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_db)
    ):
        """Delegate admin login to unified login handler."""
        return await AuthController.handle_customer_login(request, email, password, db)

    # ──────────────────────────────────────────────────────
    # LOGOUT
    # ──────────────────────────────────────────────────────

    @staticmethod
    async def handle_logout(request: Request, db: Session = Depends(get_db)):
        """Handle logout redirect to landing page."""
        user = request.session.get("user") or request.session.get("admin")
        if user:
            try:
                # Otomatis hapus riwayat percakapan chatbot saat logout (Titah Paduka Raja)
                from app.models.chat_message import ChatMessage
                db.query(ChatMessage).filter(ChatMessage.user_id == user["id"]).delete()

                log = ActivityLog(user_id=user["id"], activity="Logout", ip_address=request.client.host if request.client else None)
                db.add(log)
                db.commit()
            except Exception as e:
                logger.warning(f"Failed to record logout activity or clear chat history: {e}")
                db.rollback()
            request.session.clear()

        return RedirectResponse(url="/", status_code=303)
