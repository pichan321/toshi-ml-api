from fastapi import APIRouter
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessor import csv

router = APIRouter()

@router.get("/train/{model_id}/{col}")
async def train(model_id: str, col: str):
    df = pd.read_csv(model_id + ".csv")
    print(col)
    y = df[col]
    X = df.drop(col, axis=1) 
    variable_columns = list(X.columns)

    df, X, encoded_columns, columns, column_values = csv.process_and_encode(df, X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    # # Get the learned model parameters
    # learned_W = model.coef_[0]
    # learned_b = model.intercept_
    # print("Predict: {}", model.predict([[10]]))
    # print("Learned Weight: {:.4f}".format(learned_W))
    # print("Learned Bias: {:.4f}".format(learned_b))
    joblib.dump(model, model_id + ".pkl")

    return {
        "orginal_columns": list(df.columns),
        "column_headers": variable_columns, 
        "encoded_columns": encoded_columns, 
        "column_options": columns,
        "column_values": column_values,
        "target_column": ["charges"],
        }
