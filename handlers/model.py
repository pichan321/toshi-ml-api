from typing import Union
from fastapi import FastAPI, File, UploadFile, Form, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from handlers.upload import router
import sqlite3
from models.database import Base
from models import Users, Models
from utils import generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from fastapi import Request
import joblib
from database import connection

router = APIRouter()

@router.post("/create-model")
async def create_model(request: Request):
    db = connection.get_database()
    body = await request.json()

    new_model = Models.Models(id= str(generator.generateUuid()), name=body["name"], description=body["description"], model_type=body["model_type"], model_subtype=body["model_subtype"])
    db.add(new_model)
    db.commit()
    db.close()
    return {"body": body}

@router.get("/get-models/{type}/{subtype}")
async def get_models(type: str, subtype: str):

    db = connection.get_database()
    results = db.query(Models.Models).filter(Models.Models.model_type == type, Models.Models.model_subtype == subtype).all()
    db.close()


    return {"result": results}

@router.delete("/delete-model")
async def delete_model(request: Request):
    body = await request.json()
    db = connection.get_database()
    db.delete(db.query(Models.Models).filter(Models.Models.id == body["id"]).first())
    db.commit()
    db.flush()  # Flush the session to ensure changes are persisted
    db.close()
    return {"message": "deleted"}

@router.get("/predict/{model_id}/{value}")
def predict(model_id: str, value: float):
    db = connection.get_database()
    model = joblib.load(model_id + ".pkl")
    result = model.predict([[value]])

    db.close()
    return {"result":  str(result)}

@router.get("/startup")
async def startup_event():
    SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
    # SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    return {}