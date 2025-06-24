import io
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(file_stream, filename):
    contents = file_stream.read()
    file_like_object = io.BytesIO(contents)
    if filename.endswith(".csv"):
        df = pd.read_csv(file_like_object)
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_like_object)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    df = df.dropna(how="any")
    return df


def split_data(df, feature_columns, target_column, test_size, random_state):
    necessary_columns = feature_columns + [target_column]
    df = df.drop(columns=[x for x in list(df.columns) if x not in necessary_columns])
    oe = OrdinalEncoder()
    le = LabelEncoder()
    df.loc[:, feature_columns] = oe.fit_transform(df[feature_columns])
    X = df[feature_columns]
    y = le.fit_transform(df[target_column])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    return X_train, X_test, y_train, y_test
