"""
OTP Gateway Service — SkinCheck AI
====================================
Free implementation using Gmail SMTP (smtplib — built into Python, no extra cost).

How to get credentials:
  1. Open https://myaccount.google.com/security
  2. Enable "2-Step Verification" on your Gmail account.
  3. Search "App passwords" → Select app: Mail, Select device: Other → name it "SkinCheck"
  4. Copy the 16-character app password (spaces can be removed).
  5. Set in .env:
       MAIL_USERNAME=yourname@gmail.com
       MAIL_PASSWORD=your16charapppassword
       MAIL_FROM_NAME=SkinCheck AI
"""

import random
import string
import smtplib
import logging
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from typing import Optional
from sqlalchemy.orm import Session

from app.config import settings
from app.models.otp_token import OTPToken

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration (pulled from settings / .env)
# ─────────────────────────────────────────────
MAIL_USERNAME  = getattr(settings, "MAIL_USERNAME", "")
MAIL_PASSWORD  = getattr(settings, "MAIL_PASSWORD", "")
MAIL_FROM_NAME = getattr(settings, "MAIL_FROM_NAME", "SkinCheck AI")
OTP_EXPIRY_MINUTES = 10


# ─────────────────────────────────────────────
# Generate a secure 6-digit OTP code
# ─────────────────────────────────────────────
def _generate_code(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


# ─────────────────────────────────────────────
# Gmail SMTP sender (free, no external library)
# ─────────────────────────────────────────────
def _send_email(to_email: str, subject: str, html_body: str) -> bool:
    """
    Send an HTML email via Gmail SMTP TLS (port 587).
    Returns True if sent successfully, False otherwise.
    """
    if not MAIL_USERNAME or not MAIL_PASSWORD:
        logger.warning(
            "OTP email not sent: MAIL_USERNAME or MAIL_PASSWORD not set in .env. "
            "OTP will only be logged to console in dev mode."
        )
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{MAIL_FROM_NAME} <{MAIL_USERNAME}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.sendmail(MAIL_USERNAME, to_email, msg.as_string())

        logger.info(f"OTP email sent successfully to {to_email}")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error(
            "Gmail SMTP Authentication Failed. "
            "Make sure you are using an App Password, not your regular Gmail password. "
            "See: https://myaccount.google.com/security → App passwords"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to send OTP email to {to_email}: {e}")
        return False


# ─────────────────────────────────────────────
# HTML email templates
# ─────────────────────────────────────────────
def _make_register_email(name: str, code: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="id">
    <head><meta charset="UTF-8"></head>
    <body style="margin:0;padding:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif;">
      <table width="100%" cellpadding="0" cellspacing="0" style="padding:40px 20px;">
        <tr><td align="center">
          <table width="520" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
            <!-- Header -->
            <tr><td style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);padding:32px;text-align:center;">
              <div style="width:60px;height:60px;background:linear-gradient(135deg,#0ea5e9,#6366f1);border-radius:16px;margin:0 auto 16px;display:flex;align-items:center;justify-content:center;">
                <span style="font-size:28px;">🌿</span>
              </div>
              <h1 style="color:#ffffff;margin:0;font-size:22px;font-weight:700;">SkinCheck AI</h1>
              <p style="color:#94a3b8;margin:8px 0 0;font-size:13px;">Platform Pemantauan Kesehatan Kulit</p>
            </td></tr>
            <!-- Body -->
            <tr><td style="padding:40px 36px;">
              <h2 style="color:#0f172a;margin:0 0 8px;font-size:20px;">Halo, {name}! 👋</h2>
              <p style="color:#475569;font-size:15px;line-height:1.6;margin:0 0 28px;">
                Terima kasih sudah mendaftar di <strong>SkinCheck AI</strong>. Masukkan kode verifikasi di bawah ini untuk mengaktifkan akun Anda:
              </p>
              <!-- OTP Box -->
              <div style="background:#f0f9ff;border:2px dashed #0ea5e9;border-radius:12px;padding:28px;text-align:center;margin-bottom:28px;">
                <p style="color:#0369a1;font-size:13px;margin:0 0 12px;text-transform:uppercase;letter-spacing:2px;font-weight:600;">Kode Verifikasi</p>
                <span style="font-size:42px;font-weight:900;letter-spacing:10px;color:#0f172a;font-family:'Courier New',monospace;">{code}</span>
                <p style="color:#64748b;font-size:12px;margin:12px 0 0;">⏱ Berlaku selama <strong>{OTP_EXPIRY_MINUTES} menit</strong></p>
              </div>
              <div style="background:#fff7ed;border-left:4px solid #f97316;border-radius:8px;padding:14px 16px;margin-bottom:24px;">
                <p style="color:#9a3412;font-size:13px;margin:0;line-height:1.5;">
                  ⚠️ Jangan bagikan kode ini kepada siapapun. SkinCheck tidak akan pernah meminta kode verifikasi Anda melalui telepon atau pesan langsung.
                </p>
              </div>
              <p style="color:#94a3b8;font-size:12px;text-align:center;margin:0;">
                Jika Anda tidak mendaftar di SkinCheck AI, abaikan email ini.
              </p>
            </td></tr>
            <!-- Footer -->
            <tr><td style="background:#f8fafc;padding:20px;text-align:center;border-top:1px solid #e2e8f0;">
              <p style="color:#94a3b8;font-size:11px;margin:0;">© 2026 SkinCheck AI — Alat Indikasi Visual, Bukan Diagnosis Medis.</p>
            </td></tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """


def _make_delete_email(name: str, code: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="id">
    <head><meta charset="UTF-8"></head>
    <body style="margin:0;padding:0;background:#f1f5f9;font-family:'Segoe UI',Arial,sans-serif;">
      <table width="100%" cellpadding="0" cellspacing="0" style="padding:40px 20px;">
        <tr><td align="center">
          <table width="520" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
            <tr><td style="background:linear-gradient(135deg,#7f1d1d 0%,#450a0a 100%);padding:32px;text-align:center;">
              <span style="font-size:40px;">⚠️</span>
              <h1 style="color:#ffffff;margin:12px 0 0;font-size:22px;font-weight:700;">Konfirmasi Penghapusan Akun</h1>
              <p style="color:#fca5a5;margin:8px 0 0;font-size:13px;">SkinCheck AI — Verifikasi Keamanan</p>
            </td></tr>
            <tr><td style="padding:40px 36px;">
              <p style="color:#475569;font-size:15px;line-height:1.6;margin:0 0 20px;">
                Halo <strong>{name}</strong>, kami menerima permintaan untuk <strong>menghapus akun dan semua data</strong> Anda secara permanen dari SkinCheck AI.
              </p>
              <div style="background:#fff1f2;border:2px dashed #f43f5e;border-radius:12px;padding:28px;text-align:center;margin-bottom:28px;">
                <p style="color:#9f1239;font-size:13px;margin:0 0 12px;text-transform:uppercase;letter-spacing:2px;font-weight:600;">Kode Konfirmasi Penghapusan</p>
                <span style="font-size:42px;font-weight:900;letter-spacing:10px;color:#0f172a;font-family:'Courier New',monospace;">{code}</span>
                <p style="color:#64748b;font-size:12px;margin:12px 0 0;">⏱ Berlaku selama <strong>{OTP_EXPIRY_MINUTES} menit</strong></p>
              </div>
              <div style="background:#fef2f2;border-left:4px solid #dc2626;border-radius:8px;padding:14px 16px;">
                <p style="color:#7f1d1d;font-size:13px;margin:0;line-height:1.6;">
                  🚨 <strong>PERINGATAN:</strong> Tindakan ini tidak dapat dibatalkan. Seluruh data scan, foto, dan riwayat aktivitas akan dihapus secara permanen dari server.<br><br>
                  Jika bukan Anda yang meminta ini, segera amankan akun Anda dan abaikan email ini.
                </p>
              </div>
            </td></tr>
            <tr><td style="background:#f8fafc;padding:20px;text-align:center;border-top:1px solid #e2e8f0;">
              <p style="color:#94a3b8;font-size:11px;margin:0;">© 2026 SkinCheck AI</p>
            </td></tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """


# ─────────────────────────────────────────────
# WhatsApp OTP Gateway Sender
# ─────────────────────────────────────────────
WA_API_TOKEN   = getattr(settings, "WA_API_TOKEN", "")
WA_API_URL     = getattr(settings, "WA_API_URL", "https://api.fonnte.com/send")

def _send_whatsapp(to_number: str, message: str) -> bool:
    """
    Sends WhatsApp OTP message via gateway (e.g., Fonnte / Wablas API) or developer log.
    """
    if not to_number:
        return False

    if not WA_API_TOKEN:
        logger.info(f"[WHATSAPP GATEWAY DEV] Sent WhatsApp OTP to {to_number}: {message}")
        return False

    try:
        import urllib.request
        import json
        data = json.dumps({"target": to_number, "message": message}).encode("utf-8")
        req = urllib.request.Request(
            WA_API_URL,
            data=data,
            headers={
                "Authorization": WA_API_TOKEN,
                "Content-Type": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message to {to_number}: {e}")
        return False


# ─────────────────────────────────────────────
# Public API — used by controllers
# ─────────────────────────────────────────────

def send_registration_otp(email: str, name: str, db: Session, channel: str = "email", phone: Optional[str] = None) -> dict:
    """
    Creates a fresh 6-digit OTP for registration, saves to DB, and sends via Email or WhatsApp.
    Returns {'success': bool, 'channel': str, 'dev_code': str|None}
    """
    # Invalidate any existing unused OTPs for this email+purpose
    db.query(OTPToken).filter(
        OTPToken.email == email,
        OTPToken.purpose == "register",
        OTPToken.is_used == False
    ).delete()
    db.commit()

    code = _generate_code()
    otp = OTPToken(
        email=email,
        code=code,
        purpose="register",
        expires_at=datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    )
    db.add(otp)
    db.commit()

    if channel == "whatsapp":
        target_phone = phone or email
        msg = f"*SkinCheck AI — Kode Verifikasi WhatsApp*\n\nHalo {name},\nKode OTP Anda adalah: *{code}*\n\nKode ini berlaku selama 10 menit. Jangan bagikan kode ini kepada siapapun."
        sent = _send_whatsapp(target_phone, msg)
        logger.info(f"[WHATSAPP OTP] Verification code for {email} ({target_phone}): {code}")
        return {"success": sent, "channel": "whatsapp", "dev_code": code}

    # Default to Email channel
    sent = _send_email(email, "🌿 Kode Verifikasi SkinCheck AI", _make_register_email(name, code))

    if not sent:
        logger.info(f"[DEV MODE] Registration OTP for {email}: {code}")
        return {"success": False, "channel": "email", "dev_code": code}

    return {"success": True, "channel": "email", "dev_code": None}


def send_delete_account_otp(email: str, name: str, db: Session) -> dict:
    """Creates OTP for account deletion confirmation and emails it."""
    db.query(OTPToken).filter(
        OTPToken.email == email,
        OTPToken.purpose == "delete_account",
        OTPToken.is_used == False
    ).delete()
    db.commit()

    code = _generate_code()
    otp = OTPToken(
        email=email,
        code=code,
        purpose="delete_account",
        expires_at=datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    )
    db.add(otp)
    db.commit()

    sent = _send_email(email, "⚠️ Konfirmasi Penghapusan Akun SkinCheck", _make_delete_email(name, code))

    if not sent:
        logger.info(f"[DEV MODE] Delete-account OTP for {email}: {code}")
        return {"success": False, "dev_code": code}

    return {"success": True, "dev_code": None}


def verify_otp(email: str, code: str, purpose: str, db: Session) -> bool:
    """
    Validates the OTP code for a given email and purpose.
    Marks the token as used on success.
    Returns True if valid, False otherwise.
    """
    token = db.query(OTPToken).filter(
        OTPToken.email == email,
        OTPToken.code == code,
        OTPToken.purpose == purpose,
        OTPToken.is_used == False
    ).first()

    if not token or not token.is_valid():
        return False

    token.is_used = True
    db.commit()
    return True
