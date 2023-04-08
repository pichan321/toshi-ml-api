from sqlalchemy import create_engine, Column, Integer, String
from .database import Base

class Models(Base):
    __tablename__ = "models"
    id = Column(String(100), primary_key=True)
    name = Column(String(100), unique=False)
    description = Column(String(500))
    model_type = Column(String(100))
    model_subtype = Column(String(100))