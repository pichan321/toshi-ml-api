from fastapi import APIRouter, Request
import json
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from preprocessor import csv
import matplotlib.pyplot as plt
router = APIRouter()

@router.post("/train/{model_id}/{col}")
async def train(model_id: str, col: str, request: Request):
    body = await request.json()

    df = pd.read_csv(model_id + ".csv")
    y = df[col]
    X = df.drop(col, axis=1)
    variable_columns = list(X.columns)

    df, X, encoded_columns, columns, column_values = csv.process_and_encode(df, X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    type, subtype = body["type"], body["subtype"]

    if (type == "linear-regression"):
        if (subtype == "simple"):
            pass
        if (subtype == "polynomial"):
            pass
    
         
    
def train_regression(df, X, y):
    variable_columns = list(X.columns)

    df, X, encoded_columns, columns, column_values = csv.process_and_encode(df, X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = LinearRegression()
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

    plt.scatter(X_train, y_train)

    # Plot the regression line
    plt.plot(X_train, model.predict(X_train), color='red')

    # Add labels and title
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.title('Linear Regression')

    # Show the plot
    plt.savefig(model_id + ".png")

    return {
        "orginal_columns": list(df.columns),
        "column_headers": variable_columns, 
        "encoded_columns": encoded_columns, 
        "column_options": columns,
        "column_values": column_values,
        "target_column": ["charges"],
        }

def train_logistic():
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print("Accuracy:", score)
    pass