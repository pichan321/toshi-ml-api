from fastapi import APIRouter, Request, Response
import json
import pandas as pd
import numpy as np
import joblib

from models.Models import Models
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from preprocessor import csv
import matplotlib.pyplot as plt
from database import connection

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
router = APIRouter()

@router.post("/train/{model_id}/{col}")
async def train(model_id: str, col: str, request: Request, response: Response):
    body = await request.json()

    df = pd.read_csv(model_id + ".csv")
    
    if len(col) <= 0:
        response.status_code = 400
        return {"message": "Please choose a target column to train your model.", "code": 400, "error": "Bad Request"}

    y = df[col]
    X = df.drop(col, axis=1)

    type, subtype = body["type"], body["subtype"]

    if (type == "linear-regression"):
        if (subtype == "simple"):
            return train_regression_simple(df, X, y, model_id, subtype, target_column=col)
        if (subtype == "polynomial"):
            return train_regression_polynomial(df, X, y, model_id, subtype, body["degree"], target_column=col)
    if (type == "logistic-regression"):
        if (subtype == "binary"):
            return train_logistic(df, X, y, model_id, subtype, target_column=col)
        if (subtype == "multinomial"):
            return train_logistic(df, X, y, model_id, subtype, target_column=col)
    if (type == "decision-tree"):
            return train_decision_tree(df, X, y, model_id, subtype, target_column=col)
 
    return None

def save_model(model, model_id):
    joblib.dump(model, model_id + ".pkl")
    
def train_regression_simple(df, X, y, model_id, subtype=None, target_column=None):
    variable_columns = list(X.columns)
   
    df, X, encoded_columns, columns, column_values = csv.process_and_encode(df, X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    save_model(model, model_id)
    
    model_details = {
        'orginal_columns': list(df.columns),
        'column_headers': variable_columns, 
        'encoded_columns': encoded_columns, 
        'column_options': columns,
        'column_values': column_values,
        'target_column': target_column,
    }

    db = connection.get_database()

    model = db.query(Models).filter(Models.id == model_id).one_or_none()
    if model == None:
        return {"message": "Invalid model id"}
    model.detail = json.dumps(model_details)
    db.add(model)
    db.commit()
    db.close()
    return model_details

def train_regression_polynomial(df, X, y, model_id, subtype, degree, target_column=None):
    variable_columns = list(X.columns)
   
    df, X, encoded_columns, columns, column_values = csv.process_and_encode(df, X)


    poly = PolynomialFeatures(degree=int(degree))
    X = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_details = {
        'orginal_columns': list(df.columns),
        'column_headers': variable_columns, 
        'encoded_columns': encoded_columns, 
        'column_options': columns,
        'column_values': column_values,
        'target_column': ['charges'],
    }
    save_model(model, model_id)

    db = connection.get_database()

    model = db.query(Models).filter(Models.id == model_id).one_or_none()
    if model == None:
        return {"message": "Invalid model id"}
    model.detail = json.dumps(model_details)
    db.add(model)
    db.commit()
    db.close()
    return model_details

def train_logistic(df, X, y, model_id, subtype, target_column=None):
    variable_columns = list(X.columns)
    
    df, X, encoded_columns, columns, column_values = csv.process_and_encode(df, X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    save_model(model, model_id)

    model_details = {
        "orginal_columns": list(df.columns),
        "column_headers": variable_columns, 
        "encoded_columns": encoded_columns, 
        "column_options": columns,
        "column_values": column_values,
        "target_column": ["charges"],
    }

    db = connection.get_database()

    model = db.query(Models).filter(Models.id == model_id).one_or_none()
    if model == None:
        return {"message": "Invalid model id"}
    model.detail = str(model_details)
    db.add(model)
    db.commit()
    db.close()
    return model_details

def train_decision_tree(df, X, y, model_id, subtype, target_column=None):
    variable_columns = list(X.columns)
    
    df, X, encoded_columns, columns, column_values = csv.process_and_encode(df, X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = None
    if subtype == "standard":
        model = DecisionTreeClassifier()
    if subtype == "random-forest":
        model = RandomForestClassifier()
    if subtype == "xg-boost":
        model = xgb.XGBClassifier()

    model.fit(X, y)
    model_details = {
        'orginal_columns': list(df.columns),
        'column_headers': variable_columns, 
        'encoded_columns': encoded_columns, 
        'column_options': columns,
        'column_values': column_values,
        'target_column': ['charges'],
    }
    joblib.dump(model, model_id + ".pkl")
    db = connection.get_database()

    model = db.query(Models).filter(Models.id == model_id).one_or_none()
    if model == None:
        return {"message": "Invalid model id"}
    model.detail = json.dumps(model_details)
    db.add(model)
    db.commit()
    db.close()
    return model_details
