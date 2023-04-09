from fastapi import APIRouter
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

router = APIRouter()

@router.get("/train/{model_id}")
async def train(model_id: str):
    df = pd.read_csv(model_id + ".csv")

    # Extract input features (X) and target variable (y) from the dataframe
    X = df.iloc[:, :-1]  # Extract all columns except the last one as input features
    y = df.iloc[:, -1]  # Extract the last column as the target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    # Get the learned model parameters
    learned_W = model.coef_[0]
    learned_b = model.intercept_
    print("Predict: {}", model.predict([[10]]))
    print("Learned Weight: {:.4f}".format(learned_W))
    print("Learned Bias: {:.4f}".format(learned_b))

    joblib.dump(model, model_id + ".pkl")
    return {"column_headers": list(df.columns), "column_types": str(df.dtypes), "mean_squared_error": mse, "mean_absolute_error": mae, "r2": r2}