from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from .database import Base
from datetime import datetime

class Models(Base):
    __tablename__ = "models"
    id = Column(String(100), primary_key=True)
    name = Column(String(100), unique=False)
    description = Column(String(500))
    detail = Column(String)
    model_type = Column(String(100))
    model_subtype = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow) 