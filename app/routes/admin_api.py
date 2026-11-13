"""Admin JSON API (/api/v2/admin/*) — adaptasi fitur admin monolith.

Sumber perilaku: ``app/controllers/admin_controller.py`` (monolith MVC +
template HTML) dan ``app/services/ai_copilot_service.py``. Yang diadaptasi
di sini HANYA logika/query-nya; router HTML lama TIDAK di-mount dan
``app/main.py`` tetap JSON-only.

Adaptasi yang sengaja dipilih (bukan port mentah):
- Login admin terpisah TIDAK dibangun: login tetap terpadu, peran dibaca
  dari kolom ``users.role`` di database. Akun admin dibuat di luar
  aplikasi (oleh pemilik), bukan lewat endpoint ini.
- Foto scan TIDAK disajikan: backend API-only tidak me-mount /static dan
  file upload tidak ikut bundle deployment, jadi endpoint audit hanya
  mengembalikan metadata + skor + ``photo_served: false``. Jangan pernah
  memasang placeholder foto palsu di frontend.
- RAG sync jujur: arsitektur saat ini TIDAK punya indeks RAG terpisah
  (katalog dimuat dari database setiap permintaan chat), jadi sync-rag
  hanya melaporkan jumlah produk aktif + timestamp, tanpa klaim rebuild
  indeks.
- Kuota scan default TANPA BATAS: ``SCAN_DAILY_QUOTA`` hanya diberlakukan
  kalau bernilai angka positif. Tanpa itu, perilaku customer tidak berubah.
- Testing Hub TIDAK PERNAH dispatch: uji email hanya connect + login
  SMTP (tanpa kirim), uji WA hanya cek status device Fonnte (tanpa kirim
  pesan), uji AI hanya generate prompt singkat.
- Nilai secret TIDAK PERNAH dikembalikan: GET settings me-mask total
  kunci rahasia menjadi penanda "tersimpan"/kosong.
- Semua mutasi admin menulis satu baris ActivityLog (user_id admin,
  activity, ip_address dari request.client).
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.config import settings
from app.config.database import get_db
from app.models.activity_log import ActivityLog
from app.models.analysis_result import AnalysisResult
from app.models.chat_message import ChatMessage
from app.models.product import Product, PRODUCT_CATEGORY_VALUES, normalize_category
from app.models.skin_scan import SkinScan
from app.models.system_setting import SystemSetting
from app.models.user import User
from app.routes.json_api import _iso_utc, _log_activity, _product_dict
from app.utils.auth import get_password_hash, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/admin", tags=["Admin"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unauthorized():
    from fastapi import HTTPException

    return HTTPException(status_code=401, detail="Login required")


def _forbidden(detail: str = "Hanya admin."):
    from fastapi import HTTPException

    return HTTPException(status_code=403, detail=detail)


def _db_unavailable() -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


def _require_admin(request: Request) -> dict:
    user = request.session.get("user")
    if not isinstance(user, dict):
        raise _unauthorized()
    if user.get("role") != "admin":
        raise _forbidden()
    return user


def _admin_user_dict(user: User) -> dict:
    # password_hash TIDAK PERNAH dikembalikan.
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "status": user.status,
        "created_at": _iso_utc(user.created_at),
    }


def _page_params(page: int, per_page: int) -> tuple[int, int]:
    page = max(1, page or 1)
    per_page = min(100, max(1, per_page or 10))
    return page, per_page


def _paginate(items: list, total: int, page: int, per_page: int) -> dict:
    total_pages = (total + per_page - 1) // per_page if total else 1
    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "items": items,
    }


def _utc_day_start() -> datetime:
    now = datetime.utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _as_list(value: Optional[Union[List[str], str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AdminUserCreate(BaseModel):
    name: str = Field(..., min_length=1)
    email: EmailStr
    password: str = Field(..., min_length=6)


class AdminUserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    status: Optional[str] = None
    password: Optional[str] = None  # kosong = tidak diubah


class AdminProductIn(BaseModel):
    brand: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    description: Optional[str] = None
    ingredients: Optional[Union[List[str], str]] = None
    skin_concern: Optional[Union[List[str], str]] = None
    usage_guidance: Optional[str] = None
    shopee_link: Optional[str] = None
    tokopedia_link: Optional[str] = None
    status: Optional[str] = None


class AdminProductUpdate(BaseModel):
    brand: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    ingredients: Optional[Union[List[str], str]] = None
    skin_concern: Optional[Union[List[str], str]] = None
    usage_guidance: Optional[str] = None
    shopee_link: Optional[str] = None
    tokopedia_link: Optional[str] = None
    status: Optional[str] = None


class AutoInsertIn(BaseModel):
    name: str = Field(..., min_length=1)
    brand: Optional[str] = "SkinCheck Lab Formulations"
    category: Optional[str] = "treatment"
    description: Optional[str] = None
    ingredients: Optional[Union[List[str], str]] = None
    skin_concern: Optional[Union[List[str], str]] = None
    usage_guidance: Optional[str] = None
    shopee_link: Optional[str] = None
    tokopedia_link: Optional[str] = None


class SettingsPatchIn(BaseModel):
    settings: dict = Field(default_factory=dict)


class AdminProfileUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    old_password: Optional[str] = None
    new_password: Optional[str] = None


# ---------------------------------------------------------------------------
# Dashboard (perluasan /admin/summary lama, yang tetap dipertahankan)
# ---------------------------------------------------------------------------

@router.get("/dashboard")
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    _require_admin(request)
    try:
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.status == "active").count()
        total_scans = db.query(SkinScan).count()
        scans_today = (
            db.query(SkinScan)
            .filter(SkinScan.created_at >= _utc_day_start())
            .count()
        )
        completed = db.query(SkinScan).filter(SkinScan.status == "completed").count()
        failed = db.query(SkinScan).filter(SkinScan.status == "failed").count()
        # Success rate = completed / total scan; null kalau belum ada scan.
        success_rate = round(completed / total_scans * 100, 1) if total_scans else None
        total_products = db.query(Product).count()
        active_products = db.query(Product).filter(Product.status == "active").count()
        # Sesi chat: pesan dari user dalam 30 hari terakhir (pilihan sederhana
        # yang didokumentasikan; bukan klaim "sesi aktif real-time").
        cutoff = datetime.utcnow() - timedelta(days=30)
        chat_messages_30d = (
            db.query(ChatMessage)
            .filter(ChatMessage.role == "user", ChatMessage.created_at >= cutoff)
            .count()
        )
        chat_users_30d = (
            db.query(ChatMessage.user_id)
            .filter(ChatMessage.role == "user", ChatMessage.created_at >= cutoff)
            .distinct()
            .count()
        )
        # Verifikasi akun: siapa yang baru daftar tapi belum verifikasi
        # (pending = OTP belum lolos; awaiting_biodata = OTP lolos, biodata
        # belum lengkap), dan siapa yang baru saja menyelesaikan verifikasi
        # (jejaknya ada di activity log).
        pending_count = db.query(User).filter(User.status == "pending").count()
        awaiting_count = (
            db.query(User).filter(User.status == "awaiting_biodata").count()
        )
        pending_accounts = []
        for u in (
            db.query(User)
            .filter(User.status.in_(["pending", "awaiting_biodata"]))
            .order_by(User.created_at.desc())
            .limit(10)
            .all()
        ):
            try:
                pdata = json.loads(u.profile_data or "{}")
            except Exception:  # noqa: BLE001
                pdata = {}
            pending_accounts.append(
                {
                    "id": u.id,
                    "name": u.name,
                    "email": u.email,
                    "phone": pdata.get("phone") or "",
                    "status": u.status,
                    "created_at": _iso_utc(u.created_at),
                }
            )
        recently_verified = []
        for log in (
            db.query(ActivityLog)
            .filter(
                ActivityLog.activity.in_(
                    [
                        "Verifikasi OTP WhatsApp berhasil",
                        "Melengkapi biodata — verifikasi akun selesai",
                    ]
                )
            )
            .order_by(ActivityLog.created_at.desc())
            .limit(10)
            .all()
        ):
            owner = db.query(User).filter(User.id == log.user_id).first()
            recently_verified.append(
                {
                    "user_id": log.user_id,
                    "name": owner.name if owner else "-",
                    "email": owner.email if owner else "-",
                    "activity": log.activity,
                    "created_at": _iso_utc(log.created_at),
                }
            )
        t0 = time.time()
        db.execute(text("SELECT 1"))
        db_latency_ms = round((time.time() - t0) * 1000, 1)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin dashboard query failed: %s", e)
        return _db_unavailable()
    return JSONResponse(
        content={
            "users": total_users,
            "active_users": active_users,
            "total_scans": total_scans,
            "scans_today": scans_today,
            "completed_scans": completed,
            "failed_scans": failed,
            "success_rate": success_rate,
            "products": total_products,
            "active_products": active_products,
            "chat_messages_30d": chat_messages_30d,
            "chat_users_30d": chat_users_30d,
            "verification": {
                "pending_count": pending_count,
                "awaiting_biodata_count": awaiting_count,
                "pending_accounts": pending_accounts,
                "recently_verified": recently_verified,
            },
            "health": {
                "db": "ok",
                "db_latency_ms": db_latency_ms,
                "app_env": settings.APP_ENV,
            },
        }
    )


# ---------------------------------------------------------------------------
# Users CRUD (akun baru selalu customer; soft deactivate, tanpa hard delete)
# ---------------------------------------------------------------------------

USER_STATUSES = {"active", "inactive", "suspended", "pending", "awaiting_biodata"}


@router.get("/users")
async def admin_users(
    request: Request,
    q: str = "",
    status: str = "",
    page: int = 1,
    per_page: int = 10,
    db: Session = Depends(get_db),
):
    _require_admin(request)
    page, per_page = _page_params(page, per_page)
    try:
        query = db.query(User)
        if q:
            like = f"%{q}%"
            query = query.filter((User.name.ilike(like)) | (User.email.ilike(like)))
        if status:
            if status == "inactive":
                # Inactive = akun yang belum verifikasi (pending /
                # awaiting_biodata) + akun yang memang dinonaktifkan.
                query = query.filter(
                    User.status.in_(["inactive", "pending", "awaiting_biodata"])
                )
            else:
                query = query.filter(User.status == status)
        total = query.count()
        rows = (
            query.order_by(User.id.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("admin users query failed: %s", e)
        return _db_unavailable()
    data = _paginate([_admin_user_dict(u) for u in rows], total, page, per_page)
    data["users"] = data.pop("items")
    return JSONResponse(content=data)


@router.post("/users")
async def admin_create_user(
    request: Request, body: AdminUserCreate, db: Session = Depends(get_db)
):
    admin = _require_admin(request)
    try:
        if db.query(User).filter(User.email == body.email).first():
            return JSONResponse(
                status_code=400, content={"detail": "Email sudah terdaftar."}
            )
        user = User(
            name=body.name.strip(),
            email=body.email,
            password_hash=get_password_hash(body.password),
            role="customer",
            status="active",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin create user failed: %s", e)
        db.rollback()
        return _db_unavailable()
    _log_activity(db, request, admin["id"], f"Admin membuat akun customer: {user.email}")
    return JSONResponse(status_code=201, content=_admin_user_dict(user))


@router.patch("/users/{user_id}")
async def admin_update_user(
    request: Request, user_id: int, body: AdminUserUpdate, db: Session = Depends(get_db)
):
    admin = _require_admin(request)
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return JSONResponse(status_code=404, content={"detail": "Pengguna tidak ditemukan."})
        if body.status is not None and body.status not in USER_STATUSES:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Status tidak valid. Pilihan: {', '.join(sorted(USER_STATUSES))}."},
            )
        if body.status is not None and body.status != "active" and user.id == admin["id"]:
            return JSONResponse(
                status_code=400, content={"detail": "Admin tidak boleh menonaktifkan akunnya sendiri."}
            )
        if body.email is not None and body.email != user.email:
            if db.query(User).filter(User.email == body.email).first():
                return JSONResponse(
                    status_code=400, content={"detail": "Email sudah dipakai akun lain."}
                )
            user.email = body.email
        if body.name is not None:
            user.name = body.name.strip()
        if body.status is not None:
            user.status = body.status
        if body.password:
            user.password_hash = get_password_hash(body.password)
        db.commit()
        db.refresh(user)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin update user failed: %s", e)
        db.rollback()
        return _db_unavailable()
    _log_activity(db, request, admin["id"], f"Admin mengubah akun pengguna: {user.email}")
    return JSONResponse(content=_admin_user_dict(user))


@router.delete("/users/{user_id}")
async def admin_delete_user(request: Request, user_id: int, db: Session = Depends(get_db)):
    # Soft deactivate: status -> inactive, baris + riwayat + log tetap utuh.
    admin = _require_admin(request)
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return JSONResponse(status_code=404, content={"detail": "Pengguna tidak ditemukan."})
        if user.id == admin["id"]:
            return JSONResponse(
                status_code=400, content={"detail": "Admin tidak boleh menonaktifkan akunnya sendiri."}
            )
        user.status = "inactive"
        db.commit()
        db.refresh(user)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin deactivate user failed: %s", e)
        db.rollback()
        return _db_unavailable()
    _log_activity(db, request, admin["id"], f"Admin menonaktifkan akun: {user.email}")
    return JSONResponse(content=_admin_user_dict(user))


# ---------------------------------------------------------------------------
# Products CRUD (soft deactivate: status -> inactive, tanpa hard delete)
# ---------------------------------------------------------------------------

PRODUCT_STATUSES = {"active", "inactive"}


@router.get("/products")
async def admin_products(
    request: Request,
    q: str = "",
    status: str = "",
    page: int = 1,
    per_page: int = 10,
    db: Session = Depends(get_db),
):
    _require_admin(request)
    page, per_page = _page_params(page, per_page)
    try:
        query = db.query(Product)
        if q:
            like = f"%{q}%"
            query = query.filter((Product.name.ilike(like)) | (Product.brand.ilike(like)))
        if status:
            query = query.filter(Product.status == status)
        total = query.count()
        rows = (
            query.order_by(Product.id.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("admin products query failed: %s", e)
        return _db_unavailable()
    data = _paginate([_product_dict(p) for p in rows], total, page, per_page)
    data["products"] = data.pop("items")
    return JSONResponse(content=data)


def _apply_product_fields(product: Product, data: dict) -> None:
    if data.get("brand") is not None:
        product.brand = data["brand"].strip()
    if data.get("name") is not None:
        product.name = data["name"].strip()
    if data.get("category") is not None:
        product.category = normalize_category(data["category"])
    if data.get("description") is not None:
        product.description = data["description"]
    if data.get("ingredients") is not None:
        product.set_ingredients(_as_list(data["ingredients"]))
    if data.get("skin_concern") is not None:
        product.set_skin_concern(_as_list(data["skin_concern"]))
    if data.get("usage_guidance") is not None:
        product.usage_guidance = data["usage_guidance"]
    if data.get("shopee_link") is not None:
        product.shopee_link = data["shopee_link"] or None
    if data.get("tokopedia_link") is not None:
        product.tokopedia_link = data["tokopedia_link"] or None
    if data.get("status") is not None:
        product.status = data["status"]


@router.post("/products")
async def admin_create_product(
    request: Request, body: AdminProductIn, db: Session = Depends(get_db)
):
    admin = _require_admin(request)
    status = body.status or "active"
    if status not in PRODUCT_STATUSES:
        return JSONResponse(status_code=400, content={"detail": "Status produk tidak valid."})
    category = normalize_category(body.category)
    if category not in PRODUCT_CATEGORY_VALUES:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Kategori tidak dikenal. Pilihan: {', '.join(PRODUCT_CATEGORY_VALUES)}."},
        )
    try:
        product = Product(
            brand=body.brand.strip(),
            name=body.name.strip(),
            category=category,
            description=body.description,
            usage_guidance=body.usage_guidance,
            shopee_link=body.shopee_link or None,
            tokopedia_link=body.tokopedia_link or None,
            status=status,
        )
        product.set_ingredients(_as_list(body.ingredients))
        product.set_skin_concern(_as_list(body.skin_concern))
        db.add(product)
        db.commit()
        db.refresh(product)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin create product failed: %s", e)
        db.rollback()
        return _db_unavailable()
    _log_activity(db, request, admin["id"], f"Admin menambah produk: {product.name}")
    return JSONResponse(status_code=201, content=_product_dict(product))


@router.patch("/products/{product_id}")
async def admin_update_product(
    request: Request, product_id: int, body: AdminProductUpdate, db: Session = Depends(get_db)
):
    admin = _require_admin(request)
    if body.status is not None and body.status not in PRODUCT_STATUSES:
        return JSONResponse(status_code=400, content={"detail": "Status produk tidak valid."})
    if body.category is not None and normalize_category(body.category) not in PRODUCT_CATEGORY_VALUES:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Kategori tidak dikenal. Pilihan: {', '.join(PRODUCT_CATEGORY_VALUES)}."},
        )
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            return JSONResponse(status_code=404, content={"detail": "Produk tidak ditemukan."})
        _apply_product_fields(product, body.model_dump(exclude_unset=True))
        db.commit()
        db.refresh(product)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin update product failed: %s", e)
        db.rollback()
        return _db_unavailable()
    _log_activity(db, request, admin["id"], f"Admin mengubah produk: {product.name}")
    return JSONResponse(content=_product_dict(product))


@router.delete("/products/{product_id}")
async def admin_delete_product(
    request: Request, product_id: int, db: Session = Depends(get_db)
):
    # Soft deactivate: status -> inactive (konsisten dengan users).
    admin = _require_admin(request)
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            return JSONResponse(status_code=404, content={"detail": "Produk tidak ditemukan."})
        product.status = "inactive"
        db.commit()
        db.refresh(product)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin deactivate product failed: %s", e)
        db.rollback()
        return _db_unavailable()
    _log_activity(db, request, admin["id"], f"Admin menonaktifkan produk: {product.name}")
    return JSONResponse(content=_product_dict(product))


@router.post("/products/auto-insert")
async def admin_auto_insert_product(
    request: Request, body: AutoInsertIn, db: Session = Depends(get_db)
):
    # Satu klik approve dari AI Copilot -> produk aktif di katalog.
    admin = _require_admin(request)
    try:
        product = Product(
            brand=(body.brand or "SkinCheck Lab Formulations").strip(),
            name=body.name.strip(),
            category=normalize_category(body.category) or "serum",
            description=body.description,
            usage_guidance=body.usage_guidance,
            shopee_link=body.shopee_link or None,
            tokopedia_link=body.tokopedia_link or None,
            status="active",
        )
        product.set_ingredients(_as_list(body.ingredients))
        product.set_skin_concern(_as_list(body.skin_concern))
        db.add(product)
        db.commit()
        db.refresh(product)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin auto-insert failed: %s", e)
        db.rollback()
        return _db_unavailable()
    _log_activity(
        db, request, admin["id"], f"Admin menyetujui usulan AI ke katalog: {product.name}"
    )
    return JSONResponse(status_code=201, content=_product_dict(product))


# ---------------------------------------------------------------------------
# Audit scan (semua pengguna; foto TIDAK disajikan — lihat docstring modul)
# ---------------------------------------------------------------------------

@router.get("/scans")
async def admin_scans(
    request: Request,
    status: str = "",
    page: int = 1,
    per_page: int = 10,
    db: Session = Depends(get_db),
):
    _require_admin(request)
    page, per_page = _page_params(page, per_page)
    try:
        query = (
            db.query(SkinScan, User, AnalysisResult)
            .outerjoin(User, SkinScan.user_id == User.id)
            .outerjoin(AnalysisResult, AnalysisResult.skin_scan_id == SkinScan.id)
        )
        if status:
            query = query.filter(SkinScan.status == status)
        total = query.count()
        rows = (
            query.order_by(SkinScan.id.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("admin scans query failed: %s", e)
        return _db_unavailable()
    items = []
    for scan, owner, analysis in rows:
        from app.routes.json_api import _analysis_dict, _scan_dict

        item = _scan_dict(scan, analysis)
        item["owner"] = (
            {"id": owner.id, "name": owner.name, "email": owner.email}
            if owner
            else {"id": scan.user_id, "name": "—", "email": "—"}
        )
        # Kolom YOLO (bounding box) sengaja "—": integrasi YOLO belum
        # masuk aplikasi, jadi tidak ada koordinat yang bisa ditampilkan
        # — mengarangnya akan memalsukan hasil audit.
        item["yolo_boxes"] = None
        # Foto tidak bisa disajikan backend API-only (tanpa /static).
        item["photo_served"] = False
        items.append(item)
    data = _paginate(items, total, page, per_page)
    data["scans"] = data.pop("items")
    return JSONResponse(content=data)


# ---------------------------------------------------------------------------
# AI Copilot: tren nyata dari database + usulan dari Gemini (kalau ada key)
# ---------------------------------------------------------------------------

# Kata kunci pencocokan kondisi -> produk aktif (pencocokan substring
# case-insensitive pada daftar skin_concern produk).
_CONDITION_KEYWORDS = {
    "acne": ["acne", "jerawat"],
    "dry": ["dry", "kering"],
    "oily": ["oily", "berminyak", "minyak"],
    "dark_spots": ["dark", "flek", "hiperpigmentasi", "pigmentasi"],
    "wrinkles": ["wrinkle", "kerutan", "kerut", "garis halus"],
}


def _copilot_gaps(db: Session, trends: list) -> list:
    active = db.query(Product).filter(Product.status == "active").all()
    concerns_blob = []
    for p in active:
        concerns_blob.append(" ".join(p.get_skin_concern()).lower())
    gaps = []
    for t in trends:
        key = t["key"]
        keywords = _CONDITION_KEYWORDS.get(key, [key])
        count = sum(
            1 for blob in concerns_blob if any(k in blob for k in keywords)
        )
        avg = t["avg_score"]
        gaps.append(
            {
                "key": key,
                "label": t["label"],
                "avg_score": avg,
                "active_products": count,
                # Celah pasar: rata-rata tinggi tapi produk aktif sedikit.
                "gap": avg >= 40 and count <= 2,
            }
        )
    return gaps


@router.get("/trends")
async def admin_scan_trends(request: Request, db: Session = Depends(get_db)):
    """Tren hasil scan SELURUH pengguna (admin only).

    Grafik di halaman Tren Scan: rata-rata total 5 output VGG dari semua
    scan yang sudah dianalisis + kondisi yang paling menonjol, plus
    rekomendasi produk katalog yang cocok per kondisi. Rekomendasi hanya
    berisi nama produk dan kandungannya (tanpa deskripsi/link/harga).
    """
    _require_admin(request)
    try:
        from app.services.ai_copilot_service import AICopilotService

        trends = AICopilotService.get_platform_scan_trends(db)
        products = db.query(Product).filter(Product.status == "active").all()
        recommendations = []
        for cond in trends["conditions"]:
            matched = [
                p
                for p in products
                if AICopilotService.product_matches_condition(p, cond["key"])
            ]
            recommendations.append(
                {
                    "condition_key": cond["key"],
                    "condition_label": cond["label"],
                    "avg_score": cond["avg_score"],
                    "products": [
                        {
                            "id": p.id,
                            "brand": p.brand,
                            "name": p.name,
                            "ingredients": p.get_ingredients(),
                        }
                        for p in matched
                    ],
                }
            )
        return JSONResponse(content={**trends, "recommendations": recommendations})
    except Exception as e:  # noqa: BLE001
        logger.warning("admin trends failed: %s", e)
        return _db_unavailable()


@router.get("/chats")
async def admin_customer_chats(
    request: Request,
    q: str = "",
    page: int = 1,
    per_page: int = 10,
    db: Session = Depends(get_db),
):
    """Monitoring obrolan customer (admin only).

    Daftar customer yang pernah chat dengan asisten, diurutkan dari yang
    terakhir chat. Per customer: jumlah pesan, isi pesan terakhir + siapa
    pengirimnya + waktunya. Dipakai halaman Obrolan Customer.
    """
    _require_admin(request)
    page, per_page = _page_params(page, per_page)
    try:
        sub = (
            db.query(
                ChatMessage.user_id.label("user_id"),
                func.count(ChatMessage.id).label("total"),
                func.max(ChatMessage.created_at).label("last_at"),
            )
            .join(User, ChatMessage.user_id == User.id)
            .filter(User.role == "customer")
            .group_by(ChatMessage.user_id)
            .subquery()
        )
        base = db.query(User, sub.c.total, sub.c.last_at).join(sub, User.id == sub.c.user_id)
        if q:
            like = f"%{q}%"
            base = base.filter((User.name.ilike(like)) | (User.email.ilike(like)))
        total = base.count()
        rows = (
            base.order_by(sub.c.last_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        items = []
        for u, total_msgs, _last_at in rows:
            last = (
                db.query(ChatMessage)
                .filter(ChatMessage.user_id == u.id)
                .order_by(ChatMessage.created_at.desc())
                .first()
            )
            items.append(
                {
                    "user_id": u.id,
                    "name": u.name,
                    "email": u.email,
                    "status": u.status,
                    "total_messages": total_msgs,
                    "last_message": last.content if last else None,
                    "last_message_role": last.role if last else None,
                    "last_message_at": _iso_utc(last.created_at) if last and last.created_at else None,
                }
            )
        data = _paginate(items, total, page, per_page)
        data["chats"] = data.pop("items")
        return JSONResponse(content=data)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin chats failed: %s", e)
        return _db_unavailable()


@router.get("/chats/{user_id}")
async def admin_customer_chat_detail(
    request: Request,
    user_id: int,
    page: int = 1,
    per_page: int = 50,
    db: Session = Depends(get_db),
):
    """Riwayat chat satu customer (admin only).

    Dipakai halaman Obrolan Customer saat admin membuka percakapan satu
    customer — untuk membantu customer yang kebingungan di suatu fitur.
    Pesan dikembalikan kronologis (terlama dulu) dalam halaman terbaru.
    """
    _require_admin(request)
    page, per_page = _page_params(page, per_page)
    try:
        user = db.query(User).filter(User.id == user_id, User.role == "customer").first()
        if not user:
            return JSONResponse(status_code=404, content={"detail": "Customer tidak ditemukan."})
        total = db.query(ChatMessage).filter(ChatMessage.user_id == user_id).count()
        msgs = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        msgs = list(reversed(msgs))
        data = _paginate(
            [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "created_at": _iso_utc(m.created_at) if m.created_at else None,
                }
                for m in msgs
            ],
            total,
            page,
            per_page,
        )
        data["messages"] = data.pop("items")
        data["user"] = {"id": user.id, "name": user.name, "email": user.email, "status": user.status}
        return JSONResponse(content=data)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin chat detail failed: %s", e)
        return _db_unavailable()


@router.get("/chat-trends")
async def admin_chat_trends(
    request: Request,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    """Pertanyaan customer yang paling sering ditanyakan (admin only).

    Dikelompokkan dari isi ChatMessage yang sebenarnya (pertanyaan yang
    sama, beda kapital/tanda baca, terhitung satu). Dipakai halaman
    Obrolan Customer + chatbot admin — bahan analisis update ke depan.
    """
    _require_admin(request)
    try:
        from app.services.ai_assistant_service import top_customer_questions

        limit = max(1, min(limit or 10, 20))
        top = top_customer_questions(db, limit=limit)
        total_user_messages = (
            db.query(ChatMessage)
            .join(User, ChatMessage.user_id == User.id)
            .filter(User.role == "customer", ChatMessage.role == "user")
            .count()
        )
        total_chatters = (
            db.query(ChatMessage.user_id)
            .join(User, ChatMessage.user_id == User.id)
            .filter(User.role == "customer")
            .distinct()
            .count()
        )
        return JSONResponse(
            content={
                "top_questions": [
                    {
                        "question": t["question"],
                        "count": t["count"],
                        "last_asked_at": _iso_utc(t["last_asked_at"]) if t["last_asked_at"] else None,
                        "asked_by": t["asked_by"][:3],
                    }
                    for t in top
                ],
                "total_user_messages": total_user_messages,
                "total_chatters": total_chatters,
            }
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("admin chat trends failed: %s", e)
        return _db_unavailable()


@router.get("/ai-copilot")
async def admin_ai_copilot(request: Request, db: Session = Depends(get_db)):
    _require_admin(request)
    try:
        from app.services.ai_copilot_service import AICopilotService

        trends_data = AICopilotService.get_aggregate_skin_trends(db)
        trends = trends_data.get("trends", [])
        dominant = trends_data.get("dominant_concern", {})
        gaps = _copilot_gaps(db, trends)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin copilot trends failed: %s", e)
        return _db_unavailable()

    proposal = None
    ai_available = False
    ai_note = ""
    try:
        api_key = (SystemSetting.get_value(db, "GEMINI_API_KEY", settings.GEMINI_API_KEY) or "").strip()
        model_name = (SystemSetting.get_value(db, "AI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
        if not api_key:
            ai_note = "GEMINI_API_KEY belum disetel, sehingga usulan AI tidak tersedia. Tren dan gap analysis di bawah tetap dihitung dari data scan nyata."
        else:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            trend_lines = "\n".join(
                f"- {t['label']}: rata-rata {t['avg_score']}, {t['percentage']}% scan signifikan"
                for t in trends
            )
            prompt = (
                "Kamu adalah formulator skincare. Berdasarkan tren masalah kulit berikut:\n"
                f"{trend_lines}\n"
                f"Kondisi dominan: {dominant.get('label', '-')}. "
                "Buat SATU usulan produk skincare baru dalam format JSON saja, tanpa teks lain, "
                "dengan field: brand, name, category, description, ingredients (array string), "
                "skin_concern (array string), usage_guidance, rationale. "
                "Jangan sertakan link toko apa pun."
            )
            response = model.generate_content(
                prompt, request_options={"timeout": 30}
            )
            raw = (response.text or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`").strip()
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or not parsed.get("name"):
                raise ValueError("Respons AI bukan JSON usulan yang valid.")
            proposal = {
                "brand": parsed.get("brand") or "SkinCheck Lab Formulations",
                "name": parsed["name"],
                "category": parsed.get("category") or "treatment",
                "description": parsed.get("description") or "",
                "ingredients": _as_list(parsed.get("ingredients")),
                "skin_concern": _as_list(parsed.get("skin_concern")),
                "usage_guidance": parsed.get("usage_guidance") or "",
                "rationale": parsed.get("rationale") or "",
            }
            ai_available = True
    except ImportError:
        ai_note = "SDK Google Gemini tidak tersedia di environment ini, sehingga usulan AI tidak tersedia."
    except Exception as e:  # noqa: BLE001
        ai_note = f"Usulan AI gagal dibuat: {e}"
        proposal = None
        ai_available = False

    return JSONResponse(
        content={
            "total_analyzed_scans": trends_data.get("total_analyzed_scans", 0),
            "trends": trends,
            "dominant_concern": dominant,
            "gaps": gaps,
            "proposal": proposal,
            "ai_available": ai_available,
            "ai_note": ai_note,
            "generated_at": _iso_utc(datetime.utcnow()),
        }
    )


@router.post("/ai-copilot/sync-rag")
async def admin_sync_rag(request: Request, db: Session = Depends(get_db)):
    admin = _require_admin(request)
    try:
        active_count = db.query(Product).filter(Product.status == "active").count()
    except Exception as e:  # noqa: BLE001
        logger.warning("admin sync-rag failed: %s", e)
        return _db_unavailable()
    _log_activity(
        db, request, admin["id"], f"Admin sinkronisasi katalog RAG ({active_count} produk aktif)"
    )
    return JSONResponse(
        content={
            "status": "success",
            "active_products": active_count,
            "synced_at": _iso_utc(datetime.utcnow()),
            # Adaptasi jujur: tidak ada indeks RAG terpisah untuk dibangun
            # ulang — katalog dimuat langsung dari database setiap chat.
            "note": (
                "Arsitektur saat ini tidak memiliki indeks RAG terpisah: "
                "katalog produk dimuat langsung dari database setiap permintaan chat, "
                "sehingga tidak ada indeks yang dibangun ulang. Angka di atas adalah "
                "jumlah produk aktif yang dipakai chatbot."
            ),
        }
    )


# ---------------------------------------------------------------------------
# Settings (mask secret total; allowlist PATCH; secret kosong = tak berubah)
# ---------------------------------------------------------------------------

# (key, default, is_secret). Secret TIDAK PERNAH dikembalikan nilainya.
SETTING_DEFS = [
    ("AI_MODEL", "gemini-3.5-flash-lite", False),
    ("AI_MAX_TOKENS", "1024", False),
    ("AI_TEMPERATURE", "0.7", False),
    ("AI_SIMULATION_MODE", "off", False),
    ("SCAN_DAILY_QUOTA", "", False),  # kosong = tanpa batas (default aman)
    ("OTP_MODE", "", False),
    # REGISTRATION_OTP_MODE (2026-11-12): KUNCI INI WAJIB ADA — frontend tab
    # OTP mengirimnya bareng OTP_MODE, dan auth_controller membacanya
    # (default "dual") untuk alur OTP registrasi. Sebelumnya kunci ini
    # tidak ada di allowlist sehingga PATCH /admin/settings DITOLAK
    # (400 "Kunci di luar allowlist ditolak") dan SELURUH simpan tab OTP
    # gagal — termasuk ALLOW_SAME_PHONE_NUMBER. Jangan hapus.
    ("REGISTRATION_OTP_MODE", "dual", False),
    ("ALLOW_SAME_PHONE_NUMBER", "off", False),  # off = satu nomor WA hanya untuk satu akun (aturan register)
    ("MAINTENANCE_MODE", "off", False),
    ("VGG_THRESHOLD", "40", False),
    ("GEMINI_API_KEY", "", True),
    ("FONNTE_TOKEN", "", True),
    ("WA_API_TOKEN", "", True),
    ("MAIL_USERNAME", "", False),
    ("MAIL_PASSWORD", "", True),
    ("MAIL_FROM_NAME", "SkinCheck AI", False),
]

# Allowlist PATCH: kunci di luar daftar ini ditolak (400).
SETTINGS_ALLOWLIST = {key for key, _, _ in SETTING_DEFS}

# Kunci yang perilakunya benar-benar dibaca kode saat ini:
# - AI_MODEL / AI_MAX_TOKENS / AI_TEMPERATURE / AI_SIMULATION_MODE dibaca
#   ai_assistant_service setiap chat.
# - SCAN_DAILY_QUOTA & MAINTENANCE_MODE dibaca endpoint upload scan.
# - ALLOW_SAME_PHONE_NUMBER dibaca endpoint register (aturan nomor WA ganda).
# - OTP_MODE, VGG_THRESHOLD saat ini hanya tersimpan
#   (alur OTP masih WA-only; threshold belum dipakai inferensi) — jangan
#   mengklaim sebaliknya di frontend.


def _looks_secret(key: str) -> bool:
    k = key.upper()
    return (
        "TOKEN" in k or "PASSWORD" in k or k == "SECRET_KEY" or k.endswith("_KEY")
    )


@router.get("/settings")
async def admin_get_settings(request: Request, db: Session = Depends(get_db)):
    _require_admin(request)
    try:
        rows = db.query(SystemSetting).all()
        by_key = {r.key: r for r in rows}
        out = []
        seen = set()
        for key, default, is_secret in SETTING_DEFS:
            row = by_key.get(key)
            raw = row.value if row and row.value else None
            env_fallback = ""
            if key == "GEMINI_API_KEY":
                env_fallback = settings.GEMINI_API_KEY
            elif key in ("FONNTE_TOKEN", "WA_API_TOKEN"):
                env_fallback = settings.FONNTE_TOKEN
            elif key == "MAIL_USERNAME":
                env_fallback = settings.MAIL_USERNAME
            elif key == "MAIL_PASSWORD":
                env_fallback = settings.MAIL_PASSWORD
            configured = bool(raw or env_fallback)
            if is_secret:
                # Mask total: hanya penanda, bukan nilai.
                value = "tersimpan" if configured else ""
            else:
                value = raw if raw is not None else (env_fallback or default)
            out.append(
                {
                    "key": key,
                    "value": value,
                    "configured": configured,
                    "masked": is_secret,
                    "description": row.description if row else None,
                }
            )
            seen.add(key)
        for row in rows:
            if row.key in seen:
                continue
            secret = _looks_secret(row.key)
            out.append(
                {
                    "key": row.key,
                    "value": "tersimpan" if (secret and row.value) else ("" if secret else (row.value or "")),
                    "configured": bool(row.value),
                    "masked": secret,
                    "description": row.description,
                }
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("admin settings query failed: %s", e)
        return _db_unavailable()
    return JSONResponse(content={"settings": out})


@router.patch("/settings")
async def admin_patch_settings(
    request: Request, body: SettingsPatchIn, db: Session = Depends(get_db)
):
    admin = _require_admin(request)
    incoming = body.settings or {}
    rejected = [k for k in incoming if k not in SETTINGS_ALLOWLIST]
    if rejected:
        return JSONResponse(
            status_code=400,
            content={"detail": "Kunci di luar allowlist ditolak.", "rejected": rejected},
        )
    secret_keys = {k for k, _, s in SETTING_DEFS if s}
    # Validasi dulu semuanya (sebelum ada yang ditulis), supaya satu nilai
    # yang tidak valid tidak menyisakan setengah pengaturan sudah berubah.
    # Secret di-strip (kunci yang ditempel dengan spasi/newline di ujung
    # akan ditolak Google) — kosong tetap berarti "tidak diubah".
    normalized = {}
    for key, value in incoming.items():
        raw = "" if value is None else str(value)
        if key in secret_keys:
            if raw == "":
                continue
            normalized[key] = raw.strip()
            continue
        text = raw.strip()
        if key == "AI_MODEL":
            if not text:
                return JSONResponse(status_code=400, content={"detail": "AI_MODEL tidak boleh kosong."})
            normalized[key] = text
        elif key == "AI_TEMPERATURE":
            try:
                temp = float(text)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "AI_TEMPERATURE harus berupa angka."})
            if not 0 <= temp <= 2:
                return JSONResponse(status_code=400, content={"detail": "AI_TEMPERATURE harus di antara 0 dan 2."})
            normalized[key] = text
        elif key == "AI_MAX_TOKENS":
            try:
                tokens = int(text)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "AI_MAX_TOKENS harus berupa bilangan bulat."})
            if tokens <= 0:
                return JSONResponse(status_code=400, content={"detail": "AI_MAX_TOKENS harus lebih besar dari 0."})
            normalized[key] = text
        elif key == "ALLOW_SAME_PHONE_NUMBER":
            low = text.lower()
            if low in ("on", "true", "1", "ya"):
                normalized[key] = "on"
            elif low in ("off", "false", "0", "tidak"):
                normalized[key] = "off"
            else:
                return JSONResponse(status_code=400, content={"detail": "ALLOW_SAME_PHONE_NUMBER harus 'on' atau 'off'."})
        else:
            normalized[key] = text
    try:
        changed = []
        for key, value in normalized.items():
            SystemSetting.set_value(db, key, value)
            changed.append(key)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin settings update failed: %s", e)
        db.rollback()
        return _db_unavailable()
    if changed:
        _log_activity(
            db, request, admin["id"], f"Admin mengubah pengaturan: {', '.join(changed)}"
        )
    return JSONResponse(content={"updated": changed})


# ---------------------------------------------------------------------------
# Testing Hub (cek konektivitas saja — TIDAK PERNAH dispatch pesan/email)
# ---------------------------------------------------------------------------

@router.post("/settings/test-db")
async def admin_test_db(request: Request, db: Session = Depends(get_db)):
    admin = _require_admin(request)
    t0 = time.time()
    try:
        db.execute(text("SELECT 1"))
        counts = {
            "users": db.query(User).count(),
            "scans": db.query(SkinScan).count(),
            "products": db.query(Product).count(),
        }
        latency_ms = round((time.time() - t0) * 1000, 1)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin test-db failed: %s", e)
        return JSONResponse(
            content={"ok": False, "latency_ms": round((time.time() - t0) * 1000, 1), "detail": str(e)}
        )
    _log_activity(db, request, admin["id"], f"Admin menguji koneksi database ({latency_ms} ms)")
    return JSONResponse(
        content={
            "ok": True,
            "latency_ms": latency_ms,
            "database": settings.DB_DATABASE,
            "counts": counts,
            "detail": "Koneksi database OK.",
        }
    )


@router.post("/settings/test-ai")
async def admin_test_ai(request: Request, db: Session = Depends(get_db)):
    admin = _require_admin(request)
    t0 = time.time()
    api_key = (SystemSetting.get_value(db, "GEMINI_API_KEY", settings.GEMINI_API_KEY) or "").strip()
    model_name = (SystemSetting.get_value(db, "AI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
    if not api_key:
        return JSONResponse(
            content={
                "ok": False,
                "latency_ms": 0,
                "model": model_name,
                "detail": "GEMINI_API_KEY belum disetel.",
            }
        )
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            "Katakan 'Koneksi AI SkinCheck Berhasil' dalam satu kalimat.",
            request_options={"timeout": 30},
        )
        latency_ms = round((time.time() - t0) * 1000, 1)
        sample = (response.text or "").strip()[:500]
    except ImportError:
        return JSONResponse(
            content={
                "ok": False,
                "latency_ms": 0,
                "model": model_name,
                "detail": "SDK Google Gemini tidak tersedia di environment ini.",
            }
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            content={
                "ok": False,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "model": model_name,
                "detail": f"Gagal menghubungi Gemini: {e}",
            }
        )
    _log_activity(
        db, request, admin["id"], f"Admin menguji koneksi Gemini ({model_name}, {latency_ms} ms)"
    )
    return JSONResponse(
        content={
            "ok": True,
            "latency_ms": latency_ms,
            "model": model_name,
            "sample_output": sample,
            "detail": f"Koneksi Gemini ({model_name}) aktif ({latency_ms} ms).",
        }
    )


@router.post("/settings/test-email")
async def admin_test_email(request: Request, db: Session = Depends(get_db)):
    # Hanya connect + login SMTP. TIDAK mengirim email apa pun.
    admin = _require_admin(request)
    t0 = time.time()
    mail_user = SystemSetting.get_value(db, "MAIL_USERNAME", settings.MAIL_USERNAME)
    mail_pass = SystemSetting.get_value(db, "MAIL_PASSWORD", settings.MAIL_PASSWORD)
    if not mail_user or not mail_pass:
        return JSONResponse(
            content={
                "ok": False,
                "latency_ms": 0,
                "detail": "MAIL_USERNAME / MAIL_PASSWORD belum disetel.",
            }
        )
    try:
        import smtplib

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(mail_user, mail_pass)
        latency_ms = round((time.time() - t0) * 1000, 1)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            content={
                "ok": False,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "detail": f"Login SMTP gagal: {e}",
            }
        )
    _log_activity(db, request, admin["id"], f"Admin menguji login SMTP ({latency_ms} ms)")
    return JSONResponse(
        content={
            "ok": True,
            "latency_ms": latency_ms,
            "detail": "Login SMTP Gmail berhasil (tidak ada email yang dikirim).",
        }
    )


@router.post("/settings/test-wa")
async def admin_test_wa(request: Request, db: Session = Depends(get_db)):
    # Hanya cek status device Fonnte. TIDAK mengirim pesan WhatsApp apa pun.
    admin = _require_admin(request)
    t0 = time.time()
    token = (
        SystemSetting.get_value(db, "FONNTE_TOKEN", "")
        or SystemSetting.get_value(db, "WA_API_TOKEN", "")
        or settings.FONNTE_TOKEN
    )
    if not token:
        return JSONResponse(
            content={
                "ok": False,
                "latency_ms": 0,
                "detail": "Token Fonnte (FONNTE_TOKEN) belum disetel.",
            }
        )
    try:
        import httpx

        with httpx.Client(timeout=10) as client:
            resp = client.post(
                "https://api.fonnte.com/device",
                headers={"Authorization": token},
            )
        latency_ms = round((time.time() - t0) * 1000, 1)
        ok = resp.status_code == 200
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            content={
                "ok": False,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "detail": f"Gagal menghubungi Fonnte: {e}",
            }
        )
    _log_activity(
        db,
        request,
        admin["id"],
        f"Admin menguji koneksi Fonnte (HTTP {resp.status_code}, {latency_ms} ms)",
    )
    return JSONResponse(
        content={
            "ok": ok,
            "latency_ms": latency_ms,
            "http_status": resp.status_code,
            "detail": (
                f"Device Fonnte terverifikasi (HTTP {resp.status_code})."
                if ok
                else f"Fonnte mengembalikan HTTP {resp.status_code} — token/device belum valid."
            ),
        }
    )


# ---------------------------------------------------------------------------
# Logs & profil admin
# ---------------------------------------------------------------------------

@router.get("/logs")
async def admin_logs(
    request: Request,
    page: int = 1,
    per_page: int = 20,
    db: Session = Depends(get_db),
):
    _require_admin(request)
    page, per_page = _page_params(page, per_page)
    try:
        query = db.query(ActivityLog, User).outerjoin(User, ActivityLog.user_id == User.id)
        total = query.count()
        rows = (
            query.order_by(ActivityLog.id.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("admin logs query failed: %s", e)
        return _db_unavailable()
    items = [
        {
            "id": log.id,
            "user_id": log.user_id,
            "user_name": owner.name if owner else "—",
            "user_email": owner.email if owner else "—",
            "activity": log.activity,
            "ip_address": log.ip_address,
            "created_at": _iso_utc(log.created_at),
        }
        for log, owner in rows
    ]
    data = _paginate(items, total, page, per_page)
    data["logs"] = data.pop("items")
    return JSONResponse(content=data)


@router.get("/profile")
async def admin_profile(request: Request, db: Session = Depends(get_db)):
    admin = _require_admin(request)
    try:
        user = db.query(User).filter(User.id == admin["id"]).first()
        if not user:
            return JSONResponse(status_code=404, content={"detail": "Akun admin tidak ditemukan."})
    except Exception as e:  # noqa: BLE001
        logger.warning("admin profile query failed: %s", e)
        return _db_unavailable()
    return JSONResponse(content=_admin_user_dict(user))


@router.patch("/profile")
async def admin_update_profile(
    request: Request, body: AdminProfileUpdate, db: Session = Depends(get_db)
):
    admin = _require_admin(request)
    try:
        user = db.query(User).filter(User.id == admin["id"]).first()
        if not user:
            return JSONResponse(status_code=404, content={"detail": "Akun admin tidak ditemukan."})
        # Email TIDAK dapat diubah (aturan produk): permintaan mengubah
        # email ditolak 400, bukan diabaikan diam-diam.
        if body.email is not None and body.email != user.email:
            return JSONResponse(
                status_code=400, content={"detail": "Email tidak dapat diubah."}
            )
        if body.name is not None:
            name = body.name.strip()
            if not name:
                return JSONResponse(
                    status_code=400, content={"detail": "Nama tidak boleh kosong."}
                )
            user.name = name
        if body.new_password:
            if not body.old_password or not verify_password(
                body.old_password, user.password_hash
            ):
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Password lama salah — password tidak diubah."},
                )
            if len(body.new_password) < 6:
                return JSONResponse(
                    status_code=400, content={"detail": "Password baru minimal 6 karakter."}
                )
            user.password_hash = get_password_hash(body.new_password)
        db.commit()
        db.refresh(user)
    except Exception as e:  # noqa: BLE001
        logger.warning("admin profile update failed: %s", e)
        db.rollback()
        return _db_unavailable()
    # Segarkan sesi supaya getMe langsung menampilkan nama baru.
    sess = request.session.get("user")
    if isinstance(sess, dict):
        sess["name"] = user.name
        request.session["user"] = sess
    _log_activity(db, request, admin["id"], "Admin memperbarui profilnya")
    return JSONResponse(content=_admin_user_dict(user))
