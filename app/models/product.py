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

