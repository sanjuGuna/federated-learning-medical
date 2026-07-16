import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(filepath="archive/diabetes_data_upload.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def encode_features(df):
    df = df.copy()
    # Map Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    
    # Map Target
    if "class" in df.columns:
        df["class"] = df["class"].map({"Positive": 1, "Negative": 0})
    
    # Map remaining Yes/No columns
    yes_no_mapping = {"Yes": 1, "No": 0}
    for col in df.columns:
        if col not in ["Age", "Gender", "class"]:
            df[col] = df[col].map(yes_no_mapping)
            
    return df

def normalize_age(df, scaler=None):
    df = df.copy()
    if scaler is None:
        scaler = MinMaxScaler()
        df["Age"] = scaler.fit_transform(df[["Age"]])
    else:
        df["Age"] = scaler.transform(df[["Age"]])
    return df, scaler

def get_preprocessed_data(filepath="archive/diabetes_data_upload.csv"):
    df = load_data(filepath)
    df = encode_features(df)
    df, scaler = normalize_age(df)
    
    # Separate Features and Target
    y = df["class"].values
    X = df.drop(columns=["class"]).values
    
    return X, y, scaler, df.columns.drop("class")

if __name__ == "__main__":
    X, y, scaler, feature_names = get_preprocessed_data()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {list(feature_names)}")
