import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

def mysql_connect_args() -> dict:
    """Connect args for MySQL/MariaDB/TiDB via PyMySQL.

    Local MySQL (XAMPP etc.): returns {} unless DB_SSL=true.
    Hosted MySQL with a public endpoint (TiDB Cloud): set DB_SSL=true and
    PyMySQL is told to verify the server certificate and identity.
    No credential is ever hard-coded here; everything comes from .env.
    """
    if not getattr(settings, "DB_SSL", False):
        return {}
    args: dict = {
        "ssl_verify_cert": True,
        "ssl_verify_identity": True,
    }
    ca = getattr(settings, "DB_SSL_CA", "") or ""
    if not ca:
        try:
            import certifi

            ca = certifi.where()
        except Exception:
            ca = ""
    if ca:
        args["ssl_ca"] = ca
    return args


# Database URL configuration based on DB_CONNECTION
if settings.DB_CONNECTION == "sqlite":
    SQLALCHEMY_DATABASE_URL = f"sqlite:///./{settings.DB_DATABASE}.sqlite"
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
elif settings.DB_CONNECTION == "mysql":
    # MySQL format: mysql+pymysql://root:password@127.0.0.1:3306/skincheck_db
    # TiDB Cloud format: mysql+pymysql://<prefix>.root@<gateway>:4000/skincheck_db
    # (same dialect, different host/port/database; TLS is added via connect_args)
    password_part = f":{settings.DB_PASSWORD}" if settings.DB_PASSWORD else ""
    SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{settings.DB_USERNAME}{password_part}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_DATABASE}"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args=mysql_connect_args(),
        pool_recycle=300,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )
else:
    # Standard postgres format for future use
    SQLALCHEMY_DATABASE_URL = f"postgresql://{settings.DB_USERNAME}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_DATABASE}"
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()

def init_schema():
    """Safely adds missing columns to existing SQLite and MySQL tables (zero-downtime migration)."""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            if settings.DB_CONNECTION == "sqlite":
                # ── analysis_results: 5-condition skin scoring columns ──
                result = conn.execute(text("PRAGMA table_info(analysis_results)"))
                columns = [row[1] for row in result.fetchall()]
                if columns:
                    for col in ["dry_skin_score", "oily_skin_score", "dark_spots_score", "wrinkles_score"]:
                        if col not in columns:
                            conn.execute(text(f"ALTER TABLE analysis_results ADD COLUMN {col} FLOAT DEFAULT 0.0"))

                # ── products: marketplace link columns ──
                result2 = conn.execute(text("PRAGMA table_info(products)"))
                prod_columns = [row[1] for row in result2.fetchall()]
                if prod_columns:
                    if "shopee_link" not in prod_columns:
                        conn.execute(text("ALTER TABLE products ADD COLUMN shopee_link TEXT"))
                    if "tokopedia_link" not in prod_columns:
                        conn.execute(text("ALTER TABLE products ADD COLUMN tokopedia_link TEXT"))

                conn.commit()
            elif settings.DB_CONNECTION == "mysql":
                # ── products: marketplace link columns ──
                for col in ["shopee_link", "tokopedia_link"]:
                    try:
                        check = conn.execute(text(f"SHOW COLUMNS FROM products LIKE '{col}'")).fetchone()
                        if not check:
                            conn.execute(text(f"ALTER TABLE products ADD COLUMN {col} TEXT"))
                    except Exception:
                        pass

                # ── analysis_results: skin score columns ──
                for col in ["dry_skin_score", "oily_skin_score", "dark_spots_score", "wrinkles_score"]:
                    try:
                        check = conn.execute(text(f"SHOW COLUMNS FROM analysis_results LIKE '{col}'")).fetchone()
                        if not check:
                            conn.execute(text(f"ALTER TABLE analysis_results ADD COLUMN {col} FLOAT DEFAULT 0.0"))
                    except Exception:
                        pass

                conn.commit()
    except Exception:
        pass

# Run automatic migration on startup
init_schema()

def get_db():
    """Dependency to get the database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

