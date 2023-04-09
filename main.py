from typing import Union
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware




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

#router imports
from handlers.upload import router as upload_router
from handlers.model import router as model_router
from handlers.training import router as training_router

app = FastAPI(default_response_class=JSONResponse)

#middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

#routers
app.include_router(upload_router)
app.include_router(model_router)
app.include_router(training_router)







