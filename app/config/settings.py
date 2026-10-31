import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file
load_dotenv(os.path.join(BASE_DIR, ".env"))

class Settings:
    """Application Settings loaded from environment variables."""
    APP_NAME: str = os.getenv("APP_NAME", "SkinCheck")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_DEBUG: bool = os.getenv("APP_DEBUG", "true").lower() == "true"
    APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "default_insecure_secret_key")

    # Frontend origin(s) for CORS, comma-separated.
    # Backend-only addition for the split: the monolith did not have CORS.
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Database
    DB_CONNECTION: str = os.getenv("DB_CONNECTION", "sqlite")
    DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT: int = int(os.getenv("DB_PORT", 5432))
    DB_DATABASE: str = os.getenv("DB_DATABASE", "skincheck_db")
    DB_USERNAME: str = os.getenv("DB_USERNAME", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    # TLS for hosted MySQL (e.g. TiDB Cloud public endpoint, which requires TLS).
    # Local XAMPP/MySQL: leave DB_SSL=false. TiDB Cloud: set DB_SSL=true.
    # DB_SSL_CA is optional: path to a CA bundle. If empty and certifi is
    # installed, its bundle is used; otherwise the system default is used.
    DB_SSL: bool = os.getenv("DB_SSL", "false").lower() == "true"
    DB_SSL_CA: str = os.getenv("DB_SSL_CA", "")

    # Storage
    STORAGE_DRIVER: str = os.getenv("STORAGE_DRIVER", "local")
    STORAGE_LOCAL_PATH: str = os.getenv("STORAGE_LOCAL_PATH", "./uploads")

    # AI Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # OTP / Email Gateway (Gmail SMTP — free)
    MAIL_USERNAME: str = os.getenv("MAIL_USERNAME", "")   # Gmail address
    MAIL_PASSWORD: str = os.getenv("MAIL_PASSWORD", "")   # Gmail App Password (16 chars)
    MAIL_FROM_NAME: str = os.getenv("MAIL_FROM_NAME", "SkinCheck AI")

    # OTP via WhatsApp (Fonnte) — token diisi pemiliknya sendiri, never
    # di-commit. Service fonnte_otp membaca token per-request dari
    # system_settings (WA_API_TOKEN) dulu, lalu env ini.
    FONNTE_TOKEN: str = os.getenv("FONNTE_TOKEN", "") or os.getenv("WA_API_TOKEN", "")
    FONNTE_API_URL: str = os.getenv("FONNTE_API_URL", "") or os.getenv("WA_API_URL", "https://api.fonnte.com/send")

    # Paths
    TEMPLATE_DIR: Path = BASE_DIR / "app" / "views"
    STATIC_DIR: Path = BASE_DIR / "app" / "static"

settings = Settings()
