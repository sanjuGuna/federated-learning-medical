import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np

def load_data(filepath="archive/dermatology.data"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    # Verify row lengths
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    valid_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        cols = line.split(',')
        if len(cols) != 35:
            print(f"WARNING: Row {i+1} has unexpected number of columns: {len(cols)}. Skipping...")
        else:
            valid_lines.append(cols)
            
    # Features 1-11, 12-33 are clinical and histopathological
    # Feature 34 is Age
    # Feature 35 (index 34) is class
    columns = [f"F{i}" for i in range(1, 34)] + ["Age", "class"]
    df = pd.DataFrame(valid_lines, columns=columns)
    return df

def preprocess_dermatology(df):
    print("--- Dermatology Dataset Preprocessing ---")
    print(f"Number of samples before preprocessing: {df.shape[0]}")
    
    df = df.copy()
    
    # Replace '?' with NaN in Age
    df['Age'] = df['Age'].replace('?', np.nan)
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before preprocessing: {missing_before}")
    
    # Impute missing Age (mean imputation)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    missing_after = df.isnull().sum().sum()
    print(f"Missing values after preprocessing: {missing_after}")
    
    # Map classes from 1-6 to 0-5
    df['class'] = df['class'] - 1
    
    num_classes = df["class"].nunique()
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution:\n{df['class'].value_counts().sort_index()}")
    
    return df

def normalize_features(df, scaler=None):
    df = df.copy()
    features_to_scale = df.columns.drop("class") if "class" in df.columns else df.columns
    
    if scaler is None:
        scaler = MinMaxScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    else:
        df[features_to_scale] = scaler.transform(df[features_to_scale])
        
    return df, scaler

def get_preprocessed_data(filepath="archive/dermatology.data"):
    df = load_data(filepath)
    df = preprocess_dermatology(df)
    df, scaler = normalize_features(df)
    
    # Separate Features and Target
    y = df["class"].values
    X = df.drop(columns=["class"]).values
    
    print(f"Number of features: {X.shape[1]}")
    
    return X, y, scaler, df.columns.drop("class")

if __name__ == "__main__":
    X, y, scaler, feature_names = get_preprocessed_data()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {list(feature_names)}")
