from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .database import Base

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    profile = Column(String(500))
    email = Column(String(100))
    username = Column(String(50), unique=True)
    password = Column(String(50))

 