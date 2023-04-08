from typing import Union
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from handlers.upload import router as upload_router
from handlers.model import router as model_router


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
app = FastAPI(default_response_class=JSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)



app.include_router(upload_router)
app.include_router(model_router)

@app.get("/train/{model_id}")
async def train(model_id: str):

    df = pd.read_csv(model_id + ".csv")

    # Extract input features (X) and target variable (y) from the dataframe
    X = df.iloc[:, :-1]  # Extract all columns except the last one as input features
    y = df.iloc[:, -1]  # Extract the last column as the target variable

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Get the learned model parameters
    learned_W = model.coef_[0]
    learned_b = model.intercept_
    print("Predict: {}", model.predict([[10]]))
    print("Learned Weight: {:.4f}".format(learned_W))
    print("Learned Bias: {:.4f}".format(learned_b))

    joblib.dump(model, model_id + ".pkl")
    return {}


@app.post("/users")
async def create_user():
    user = Users.Users(username="vattnaa", password="12312312", email="pchan")
    # db.add(user)
    # db.commit()
    # db.refresh(user)
    return {"user": user}

# Query users
@app.get("/users")
async def get_users(skip: int = 0, limit: int = 10):
    users = db.query(Users.Users).all()
    return users




