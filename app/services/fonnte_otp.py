"""OTP via WhatsApp (Fonnte) untuk API v2.

Dipakai oleh ``POST /api/v2/auth/otp/request`` dan
``POST /api/v2/auth/otp/verify``. Sengaja terpisah dari legacy
``otp_service.py`` (alurnya email + session HTML di controller lama):
service ini tidak mengubah alur register /api/v2 yang sudah ada
(register tetap langsung, tanpa gate OTP), dan tidak pernah
mengembalikan kode OTP lewat respons API — kode hanya dikirim lewat
WhatsApp ke nomor yang diminta.

Token Fonnte TIDAK ada di kode ini dan tidak boleh di-commit:
  1. ``system_settings`` key ``WA_API_TOKEN`` (diisi admin lewat
     dashboard, kalau tabel itu ada), lalu
  2. environment variable ``FONNTE_TOKEN`` (atau ``WA_API_TOKEN``),
  3. URL: ``system_settings`` ``WA_API_URL`` / env ``FONNTE_API_URL``
     / ``WA_API_URL``, default ``https://api.fonnte.com/send``.

Fonnte send API (dokumen resmi Fonnte): POST ke URL di atas, header
``Authorization: <token>`` (token mentah, bukan Bearer), body
``target`` + ``message`` + ``countryCode``. Contoh dokumen memakai
target format lokal (``08123456789``) dengan ``countryCode: 62``,
jadi nomor dinormalisasi dulu ke format 62 untuk kunci di database,
lalu diubah lagi ke format 0 saat dikirim ke Fonnte.

Penyimpanan: memakai tabel ``otp_tokens`` yang sudah ada di
``skincheck_setup.sql`` (tidak perlu migrasi). Kolom ``email`` di
tabel itu dipakai sebagai identifier — untuk OTP WhatsApp isinya
nomor yang sudah dinormalisasi (62...), dengan ``purpose='whatsapp'``.
Kode berlaku 10 menit, sekali pakai, dan minta ulang dibatasi 60 detik
per nomor supaya tidak bisa di-spam (dan tidak memboroskan kuota
Fonnte).
"""

import logging
import os
import re
import secrets
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models.otp_token import OTPToken

logger = logging.getLogger(__name__)

OTP_EXPIRY_MINUTES = 10
RESEND_COOLDOWN_SECONDS = 60
FONNTE_DEFAULT_URL = "https://api.fonnte.com/send"


class InvalidPhoneError(ValueError):
    pass


class TooSoonError(ValueError):
    pass


class FonnteNotConfiguredError(RuntimeError):
    pass


class FonnteSendError(RuntimeError):
    pass


def normalize_phone(raw: str) -> str:
    """Normalisasi nomor Indonesia ke format 62... (tanpa +).

    Terima: 0812..., +62812..., 62812... (spasi/strip diabaikan).
    Tolak nomor lain — OTP WhatsApp di sini hanya untuk nomor Indonesia.
    """
    if not raw:
        raise InvalidPhoneError("Nomor WhatsApp wajib diisi.")
    digits = re.sub(r"[^\d]", "", raw.strip())
    if raw.strip().startswith("+"):
        digits = raw.strip()[1:]
        digits = re.sub(r"[^\d]", "", digits)
    if digits.startswith("0"):
        digits = "62" + digits[1:]
    if not re.fullmatch(r"62\d{9,13}", digits):
        raise InvalidPhoneError(
            "Nomor WhatsApp tidak valid. Gunakan nomor Indonesia, "
            "misalnya 081234567890 atau 6281234567890."
        )
    return digits


def mask_phone(phone62: str) -> str:
    """62********890 — nomor tidak pernah ditampilkan utuh di respons."""
    if len(phone62) <= 5:
        return phone62
    return phone62[:2] + "*" * (len(phone62) - 5) + phone62[-3:]


def _fonnte_config(db: Session):
    """Token + URL Fonnte: system_settings dulu, lalu env, lalu default URL."""
    token = None
    url = None
    try:
        from app.models.system_setting import SystemSetting

        token = SystemSetting.get_value(db, "WA_API_TOKEN")
        url = SystemSetting.get_value(db, "WA_API_URL")
    except Exception as e:  # noqa: BLE001 - tabel boleh belum ada di env tertentu
        logger.debug("system_settings tidak bisa dibaca untuk Fonnte: %s", e)
    token = (token or "").strip() or os.getenv("FONNTE_TOKEN", "").strip() or os.getenv("WA_API_TOKEN", "").strip()
    url = (url or "").strip() or os.getenv("FONNTE_API_URL", "").strip() or os.getenv("WA_API_URL", "").strip() or FONNTE_DEFAULT_URL
    return token, url


def _send_whatsapp(phone62: str, message: str, db: Session) -> None:
    token, url = _fonnte_config(db)
    if not token:
        raise FonnteNotConfiguredError(
            "Token Fonnte belum dikonfigurasi."
        )
    # Format target sesuai contoh dokumen Fonnte: 08... + countryCode 62.
    target = "0" + phone62[2:]
    try:
        import httpx

        resp = httpx.post(
            url,
            json={"target": target, "message": message, "countryCode": "62"},
            headers={"Authorization": token},
            timeout=15,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Fonnte request gagal untuk %s: %s", mask_phone(phone62), e)
        raise FonnteSendError("Gagal menghubungi Fonnte.") from e
    ok = False
    reason = ""
    try:
        data = resp.json()
        ok = bool(data.get("status"))
        reason = str(data.get("reason") or data.get("message") or "")
    except Exception:  # noqa: BLE001
        ok = resp.status_code == 200
    if resp.status_code != 200 or not ok:
        logger.warning(
            "Fonnte menolak kirim ke %s (HTTP %s, reason: %s)",
            mask_phone(phone62), resp.status_code, reason,
        )
        # Alasan Fonnte diteruskan ke user (bukan token) supaya diagnosa
        # tidak lagi sekadar "gagal" — misalnya perangkat tidak terhubung
        # atau token salah akan terbaca dari pesan ini.
        detail = "Fonnte menolak pengiriman WhatsApp."
        if reason:
            detail += f" Alasan dari Fonnte: {reason[:200]}"
        raise FonnteSendError(detail)
    logger.info("OTP WhatsApp terkirim via Fonnte ke %s", mask_phone(phone62))


def request_whatsapp_otp(db: Session, phone_raw: str, name: str = "") -> dict:
    """Buat OTP, kirim via Fonnte. Respons TIDAK pernah berisi kode."""
    phone = normalize_phone(phone_raw)
    # Cek konfigurasi dulu, sebelum OTP dibuat — kalau token belum ada,
    # jangan sampai ada OTP mengambang yang tidak pernah terkirim.
    token, _url = _fonnte_config(db)
    if not token:
        raise FonnteNotConfiguredError(
            "Token Fonnte belum dikonfigurasi."
        )
    now = datetime.utcnow()
    latest = (
        db.query(OTPToken)
        .filter(OTPToken.email == phone, OTPToken.purpose == "whatsapp", OTPToken.is_used == False)  # noqa: E712
        .order_by(OTPToken.created_at.desc())
        .first()
    )
    if latest and latest.created_at and (now - latest.created_at).total_seconds() < RESEND_COOLDOWN_SECONDS:
        raise TooSoonError("Kode baru saja dikirim. Tunggu 60 detik sebelum minta ulang.")
    # Hanguskan OTP lama yang belum dipakai untuk nomor ini.
    db.query(OTPToken).filter(
        OTPToken.email == phone, OTPToken.purpose == "whatsapp", OTPToken.is_used == False  # noqa: E712
    ).delete()
    db.commit()

    code = "".join(secrets.choice("0123456789") for _ in range(6))
    otp = OTPToken(
        email=phone,
        code=code,
        purpose="whatsapp",
        expires_at=now + timedelta(minutes=OTP_EXPIRY_MINUTES),
    )
    db.add(otp)
    db.commit()

    greeting = f"Halo {name.strip()}, " if name and name.strip() else "Halo, "
    message = (
        f"SkinCheck — Kode Verifikasi WhatsApp\n\n{greeting}"
        f"Kode OTP kamu adalah: *{code}*\n\n"
        f"Kode ini berlaku selama {OTP_EXPIRY_MINUTES} menit. "
        "Jangan bagikan kode ini ke siapa pun."
    )
    try:
        _send_whatsapp(phone, message, db)
    except Exception:
        # Kirim gagal: hapus OTP-nya supaya tidak ada kode yang bisa
        # ditebak/dipakai padahal user tidak pernah menerimanya.
        db.query(OTPToken).filter(OTPToken.id == otp.id).delete()
        db.commit()
        raise
    return {
        "channel": "whatsapp",
        "phone_masked": mask_phone(phone),
        "expires_in_minutes": OTP_EXPIRY_MINUTES,
        "cooldown_seconds": RESEND_COOLDOWN_SECONDS,
    }


def verify_whatsapp_otp(db: Session, phone_raw: str, code_raw: str) -> bool:
    """Verifikasi kode; True hanya kalau cocok, belum dipakai, belum kedaluwarsa."""
    phone = normalize_phone(phone_raw)
    code = (code_raw or "").strip()
    if not re.fullmatch(r"\d{6}", code):
        return False
    token = (
        db.query(OTPToken)
        .filter(
            OTPToken.email == phone,
            OTPToken.code == code,
            OTPToken.purpose == "whatsapp",
            OTPToken.is_used == False,  # noqa: E712
        )
        .first()
    )
    if not token or not token.is_valid():
        return False
    token.is_used = True
    db.commit()
    return True
