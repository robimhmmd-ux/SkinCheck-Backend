import json
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Text
from app.config.database import Base

# ==========================================
# SQLAlchemy ORM Model
# ==========================================
class Product(Base):
    """Product model for the internal Product Catalog."""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    brand = Column(String(100), nullable=False)
    name = Column(String(150), nullable=False)
    category = Column(String(100), nullable=False) # cleanser, toner, serum, etc.
    description = Column(Text, nullable=True)
    ingredients = Column(Text, nullable=True)      # JSON list
    skin_concern = Column(Text, nullable=True)     # JSON list
    usage_guidance = Column(Text, nullable=True)
    shopee_link = Column(Text, nullable=True)      # Link produk di Shopee
    tokopedia_link = Column(Text, nullable=True)   # Link produk di Tokopedia
    status = Column(String(50), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def set_ingredients(self, data: List[str]):
        self.ingredients = json.dumps(data)
        
    def get_ingredients(self) -> List[str]:
        if self.ingredients:
            try:
                data = json.loads(self.ingredients)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
            return [i.strip() for i in self.ingredients.split(",") if i.strip()]
        return []

    def set_skin_concern(self, data: List[str]):
        self.skin_concern = json.dumps(data)
        
    def get_skin_concern(self) -> List[str]:
        if self.skin_concern:
            try:
                data = json.loads(self.skin_concern)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
            return [c.strip() for c in self.skin_concern.split(",") if c.strip()]
        return []


# ==========================================
# Kategori jenis produk (canonical)
# ==========================================
# Satu-satunya daftar kategori yang dipakai marketplace (user) dan form
# produk (admin). Nilai disimpan lowercase; label tampil di UI. Kategori
# lama ("cleanser") dinormalisasi ke yang canonical ("face wash") saat
# dibaca/ditulis, jadi produk lama tetap masuk filter yang benar.
PRODUCT_CATEGORIES = [
    ("face wash", "Face Wash"),
    ("toner", "Toner"),
    ("serum", "Serum"),
    ("moisturizer", "Moisturizer"),
    ("sunscreen", "Sunscreen"),
    ("mask", "Masker"),
    ("exfoliator", "Eksfoliator"),
]
PRODUCT_CATEGORY_VALUES = [value for value, _ in PRODUCT_CATEGORIES]
PRODUCT_CATEGORY_LABELS = dict(PRODUCT_CATEGORIES)

LEGACY_CATEGORY_MAP = {
    "cleanser": "face wash",
    "facewash": "face wash",
    "face-wash": "face wash",
    "pembersih": "face wash",
    "sunblock": "sunscreen",
    "sun screen": "sunscreen",
    "pelembap": "moisturizer",
}


def normalize_category(value) -> str:
    """Kembalikan nilai kategori canonical (lowercase), atau '' bila kosong."""
    if not value:
        return ""
    key = str(value).strip().lower()
    return LEGACY_CATEGORY_MAP.get(key, key)


def category_label(value) -> str:
    """Label tampil untuk sebuah nilai kategori (legacy ikut dinormalisasi)."""
    return PRODUCT_CATEGORY_LABELS.get(normalize_category(value), (value or "Skincare"))


# ==========================================
# Pydantic Schemas (for validation)
# ==========================================
class ProductSchema(BaseModel):
    id: int
    brand: str
    name: str
    category: str
    description: Optional[str] = None
    shopee_link: Optional[str] = None
    tokopedia_link: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

