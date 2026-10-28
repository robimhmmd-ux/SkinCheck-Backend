from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.config.database import Base

class SystemSetting(Base):
    """Key-value storage for dynamic system settings and API configurations."""
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, index=True, nullable=False)
    value = Column(Text, nullable=True)
    description = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @classmethod
    def get_value(cls, db, key: str, default: str = None) -> str:
        setting = db.query(cls).filter(cls.key == key).first()
        return setting.value if setting and setting.value else default

    @classmethod
    def set_value(cls, db, key: str, value: str, description: str = None):
        setting = db.query(cls).filter(cls.key == key).first()
        if not setting:
            setting = cls(key=key, value=value, description=description)
            db.add(setting)
        else:
            setting.value = value
            if description:
                setting.description = description
        db.commit()
        return setting
