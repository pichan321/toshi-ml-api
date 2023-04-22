from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def process_and_encode(df, X):
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_features))
    X = pd.concat([X.drop(columns=categorical_features), X_encoded_df], axis=1)

    columns = {}
    column_values = {}
    for column in list(df.columns):
        if column not in categorical_features: 
            columns[column] = []
            column_values[column] = []
            continue

        encoder = OneHotEncoder()
        encoder.fit(df[[column]])
        columns[column] = encoder.categories_[0].tolist()

        for category in columns[column]:
            column_values[column +  "_" + category] = []

    return df, X, list(X.columns), columns, column_values