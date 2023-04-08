from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from handlers import upload
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
app = FastAPI(default_response_class=JSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
db = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = db()
conn = None
try:
    conn = sqlite3.connect('mydb.sqlite')  # Replace with your desired SQLite database file name

except sqlite3.Error as e:
    print(e)


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    with open(file.filename, "wb") as f:
        f.write(contents)
    return {"filename": file.filename, "file_size": len(contents)}


@app.get("/train")
async def train():

    df = pd.read_csv('Salary_Data.csv')

    # Extract input features (X) and target variable (y) from the dataframe
    X = df['YearsExperience'].values.reshape(-1, 1)  # Reshape X to a 2D array
    y = df['Salary'].values

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
    return {}

@app.get("/startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)

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

@app.post("/create-model")
async def create_model(request: Request):
    body = await request.json()

    new_model = Models.Models(id= str(generator.generateUuid()), name="Hello", description="None", model_type="linear-regression")
    db.add(new_model)
    db.commit()
    return {"body": body}

@app.get("/get-models")
async def get_models():
    return {"result": db.query(Models.Models).all()}