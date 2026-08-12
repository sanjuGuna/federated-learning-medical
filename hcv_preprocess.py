import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

def load_data(filepath="archive/hcvdat0.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def preprocess_hcv(df):
    print("--- HCV Dataset Preprocessing ---")
    print(f"Number of samples before preprocessing: {df.shape[0]}")
    
    # Check missing values before
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before preprocessing: {missing_before}")

    df = df.copy()

    # 1. Remove "Unnamed: 0"
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 2. Encode Sex
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"m": 0, "f": 1})

    # 3. Handle missing values (Mean imputation for numerical columns)
    num_cols = ["ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Check missing values after
    missing_after = df.isnull().sum().sum()
    print(f"Missing values after preprocessing: {missing_after}")

    # 4. Encode Category (Target)
    le = LabelEncoder()
    df["Category"] = le.fit_transform(df["Category"])
    
    num_classes = df["Category"].nunique()
    class_dist = df["Category"].value_counts().to_dict()
    
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution:\n{df['Category'].value_counts().sort_index()}")
    
    return df, le

def normalize_features(df, scaler=None):
    df = df.copy()
    features_to_scale = ["Age"] + ["ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]
    
    # Filter only columns that exist
    features_to_scale = [f for f in features_to_scale if f in df.columns]
    
    if scaler is None:
        scaler = MinMaxScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    else:
        df[features_to_scale] = scaler.transform(df[features_to_scale])
        
    return df, scaler

def get_preprocessed_data(filepath="archive/hcvdat0.csv"):
    df = load_data(filepath)
    df, le = preprocess_hcv(df)
    df, scaler = normalize_features(df)
    
    # Separate Features and Target
    y = df["Category"].values
    X = df.drop(columns=["Category"]).values
    
    print(f"Number of features: {X.shape[1]}")
    
    return X, y, scaler, df.columns.drop("Category")

if __name__ == "__main__":
    X, y, scaler, feature_names = get_preprocessed_data()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {list(feature_names)}")
