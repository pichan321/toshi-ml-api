from typing import Union
from fastapi import FastAPI, File, UploadFile, Form, APIRouter, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from handlers.upload import router
import sqlite3
from models.database import Base

from models import Users, Models
from utils import generator
import json
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from fastapi import Request
import joblib
from database import connection

from sklearn.preprocessing import OneHotEncoder

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

@router.get("/model/details/{model_id}")
async def get_model_details(model_id: str, response: Response):
    db = connection.get_database()
    
    model_details = db.query(Models.Models.detail).filter(Models.Models.id == model_id).one_or_none()
    if model_details == None:
        response.status_code = 400
        return {"message": "Invalid model id", "code": 400}
    
    db.close()
    return json.loads(model_details["detail"])

@router.delete("/delete-model")
async def delete_model(request: Request):
    body = await request.json()
    db = connection.get_database()
    db.delete(db.query(Models.Models).filter(Models.Models.id == body["id"]).first())
    db.commit()
    db.flush()  # Flush the session to ensure changes are persisted
    db.close()
    return {"message": "deleted"}

@router.post("/predict/{model_id}")
async def predict(model_id: str, request: Request):

    body = await request.json()
    db = connection.get_database()
    model = joblib.load(model_id + ".pkl")

    data = {}
    # for column in body["encoded_columns"]:
    #     data[column] = body["values"][column]

    for column in body["encoded_columns"]:
        data[column] = body["values"][column]
    new_data = pd.DataFrame(data)
    # Perform one-hot encoding on the new data within the DataFrame
    # categorical_features = ['sex', "smoker", "region"]  # List of categorical feature column names
    # encoder = OneHotEncoder()
    # X_encoded = encoder.fit_transform(new_data[categorical_features])
    # X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_features))
    # new_data = pd.concat([new_data.drop(columns=categorical_features), X_encoded_df], axis=1)


    # X_encoded = encoder.fit_transform(X[categorical_features])
    # X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_features))
    # X = pd.concat([X.drop(columns=categorical_features), X_encoded_df], axis=1)
    # Make predictions on the new data
    X_new = new_data  # Assuming 'X_new' is the feature matrix of the new data
    y_pred = model.predict(new_data)  # Predict using the trained model
    db.close()
    return {"result":  str(y_pred), "code": 200}

@router.get("/model/predict/help/{model_id}")
async def get_prediction_help(model_id: str, response: Response):
    db = connection.get_database()

    model_detail = db.query(Models.Models.detail).filter(Models.Models.id == model_id).one_or_none()
    if not model_detail:
        response.status_code = 404
        return {"message": "Unable to retrieve model detail.", "code": 404}
    
    model_detail_json = json.loads(model_detail[0])
    column_headers = model_detail_json["column_headers"]
    for column in column_headers:
        if len(model_detail_json["column_options"][column]) > 0:
            pass

    return {"prediction_body": model_detail_json}

@router.get("/startup")
async def startup_event():
    SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
    # SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    return {}