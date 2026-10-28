from app.config import settings
from app.config.database import engine, Base, SessionLocal, mysql_connect_args
from app.models.user import User
from app.models.product import Product
from app.models.skin_scan import SkinScan
from app.models.analysis_result import AnalysisResult
from app.models.activity_log import ActivityLog
from app.models.otp_token import OTPToken
from app.models.system_setting import SystemSetting
from app.models.chat_message import ChatMessage
from app.utils.auth import get_password_hash
from sqlalchemy import create_engine, text

def ensure_database_exists():
    if settings.DB_CONNECTION == "mysql":
        password_part = f":{settings.DB_PASSWORD}" if settings.DB_PASSWORD else ""
        server_url = f"mysql+pymysql://{settings.DB_USERNAME}{password_part}@{settings.DB_HOST}:{settings.DB_PORT}"
        try:
            # Same TLS handling as the main engine (needed for TiDB Cloud).
            # If this step fails (e.g. the database was already created in
            # the TiDB Cloud console), init_db() still continues below and
            # creates the tables with Base.metadata.create_all().
            temp_engine = create_engine(server_url, connect_args=mysql_connect_args())
            with temp_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{settings.DB_DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"))
                conn.commit()
            print(f"MySQL Database '{settings.DB_DATABASE}' is ready.")
        except Exception as e:
            print(f"Catatan: Pastikan database MySQL sudah aktif / sudah dibuat (di TiDB Cloud, buat dulu dari console). Detail: ({e})")

def init_db():
    ensure_database_exists()
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    # NOTE (split backend): demo accounts are intentionally NOT created here.
    # The monolith's setup_db.py created admin/customer demo accounts with
    # hard-coded passwords, which must not ship in a public repo used for an
    # online demo. Create accounts via POST /api/v2/auth/register instead,
    # then promote one to admin with:
    #   UPDATE `users` SET `role` = 'admin' WHERE `email` = 'your@email';
    db.close()
    print("Database initialization complete.")

if __name__ == "__main__":
    init_db()
