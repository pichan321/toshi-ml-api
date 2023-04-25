from fastapi import APIRouter, Request
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from preprocessor import csv
import matplotlib.pyplot as plt
router = APIRouter()

@router.post("/train/{model_id}/{col}")
async def train(model_id: str, col: str, request: Request):
    body = await request.json()

    df = pd.read_csv(model_id + ".csv")
    y = df[col]
    X = df.drop(col, axis=1)

    type, subtype = body["type"], body["subtype"]

    if (type == "linear-regression"):
        if (subtype == "simple"):
            return train_regression(df, X, y, model_id, subtype)
        if (subtype == "polynomial"):
            return train_regression(df, X, y, model_id, subtype)
    if (type == "logistic-regression"):
        if (subtype == "binary"):
            return train_logistic(df, X, y, model_id, subtype)
        if (subtype == "multinomial"):
            return train_logistic(df, X, y, model_id, subtype)
    return None
         
    
def train_regression(df, X, y, model_id, subtype):
    variable_columns = list(X.columns)
   
    df, X, encoded_columns, columns, column_values = csv.process_and_encode(df, X)

    if subtype == "polynomial":
        poly = PolynomialFeatures(degree=2)
        X = poly.fit_transform(X)

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


    if subtype == "simple":
        plt.scatter(X_train, y_train)

        # Plot the regression line
        plt.plot(X_train, model.predict(X_train), color='red')

        # Add labels and title
        plt.xlabel('Independent Variable')
        plt.ylabel('Dependent Variable')
        plt.title('Linear Regression')

        # Show the plot
        plt.savefig(model_id + ".png")
    elif subtype == "multiple":
        fig, axs = plt.subplots(nrows=1, ncols=X_train.shape[1], figsize=(20, 5))

        for i, ax in enumerate(axs):
            ax.scatter(X_train[:, i], y_train)
            ax.plot(X_train[:, i], model.coef_[i] * X_train[:, i] + model.intercept_, color='red')
            ax.set_xlabel(f'Independent Variable {i+1}')
            ax.set_ylabel('Dependent Variable')
            ax.set_title(f'Linear Regression for Independent Variable {i+1}')

        plt.savefig(model_id + ".png")
    else: 
        plt.scatter(X_train, y_train)

        # Calculate the coefficients of the polynomial regression curve
        coefficients = np.polyfit(X_train[:, 0], y_train, deg=3)

        # Create a polynomial regression function
        poly_func = np.poly1d(coefficients)

        # Create a range of values for the independent variable
        x_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)

        # Plot the polynomial regression curve
        plt.plot(x_range, poly_func(x_range), color='red')

        # Add labels and title
        plt.xlabel('Independent Variable')
        plt.ylabel('Dependent Variable')
        plt.title('Polynomial Regression')

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

def train_logistic(df, X, y, model_id, subtype):
    variable_columns = list(X.columns)

    df, X, encoded_columns, columns, column_values = csv.process_and_encode(df, X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    joblib.dump(model, model_id + ".pkl")

    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    # Create a meshgrid of points to plot the decision boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict the class labels for the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Reshape the predictions to the shape of the meshgrid
    Z = Z.reshape(xx.shape)

    # Create a filled contour plot of the class predictions
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

    # Plot the training data
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolor='black')

    # Set the plot limits and labels
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')

    # Show the plot
    plt.show()
    print("Accuracy:", score)
    return {
        "orginal_columns": list(df.columns),
        "column_headers": variable_columns, 
        "encoded_columns": encoded_columns, 
        "column_options": columns,
        "column_values": column_values,
        "target_column": ["charges"],
        }