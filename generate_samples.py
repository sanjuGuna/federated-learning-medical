import json
import torch
import numpy as np
import pandas as pd
from graph_builder import create_pyg_data
from model import HybridClassifier
import os

def get_samples(dataset_name):
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load original data and preprocess
    if dataset_name == "diabetes":
        from preprocess import load_data, encode_features, normalize_age, get_preprocessed_data
        filepath = "archive/diabetes_data_upload.csv"
        df_orig = load_data(filepath)
        df_orig_encoded = encode_features(df_orig.copy())
        df_orig_norm, scaler = normalize_age(df_orig_encoded.copy())
        
        y_orig = df_orig_norm["class"].values
        X_orig = df_orig_norm.drop(columns=["class"]).values
        feature_names = df_orig_norm.columns.drop("class").tolist()
        
    elif dataset_name == "hcv":
        from hcv_preprocess import load_data, preprocess_hcv, normalize_features
        filepath = "archive/hcvdat0.csv"
        df_orig = load_data(filepath)
        df_orig_encoded, le = preprocess_hcv(df_orig.copy())
        df_orig_norm, scaler = normalize_features(df_orig_encoded.copy())
        
        y_orig = df_orig_norm["Category"].values
        X_orig = df_orig_norm.drop(columns=["Category"]).values
        feature_names = df_orig_norm.columns.drop("Category").tolist()
        
    elif dataset_name == "dermatology":
        from dermatology_preprocess import load_data, preprocess_dermatology, normalize_features
        filepath = "archive/dermatology.data"
        df_orig = load_data(filepath)
        df_orig_encoded = preprocess_dermatology(df_orig.copy())
        df_orig_norm, scaler = normalize_features(df_orig_encoded.copy())
        
        y_orig = df_orig_norm["class"].values
        X_orig = df_orig_norm.drop(columns=["class"]).values
        feature_names = df_orig_norm.columns.drop("class").tolist()
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data = create_pyg_data(X_orig, y_orig, k=5)
    
    # 4. Load Model
    in_channels = X_orig.shape[1]
    num_classes = len(np.unique(y_orig))
    
    model_path = f"federated_models/{dataset_name}/best_global_model.pt"
    if os.path.exists(model_path):
        model = HybridClassifier(in_channels=in_channels, num_classes=num_classes, mode='gat')
    else:
        model_path = f"saved_models/{dataset_name}/hybrid.pth"
        if not os.path.exists(model_path) and dataset_name == 'diabetes':
            model_path = "saved_models/hybrid.pth"
        model = HybridClassifier(in_channels=in_channels, num_classes=num_classes, mode='hybrid')
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 5. Predict
    with torch.no_grad():
        out = model(data.x, data.edge_index) if model.mode != 'rdbn' else model(data.x)
        if isinstance(out, tuple):
            out = out[0]
        preds = out.argmax(dim=1).numpy()
    
    samples = {}
    
    # We want a sample from class 0 and a sample from another class
    for cls in np.unique(preds):
        idx = np.where(preds == cls)[0]
        if len(idx) > 0:
            sample_idx = idx[0]
            # Get original data for this sample (before normalization/encoding if possible)
            # Actually, to predict via the API, the user provides raw data or similar. 
            # The API preprocesses it. 
            # Let's get the original un-preprocessed row from df_orig (but maybe need to clean a bit)
            sample_row = df_orig.iloc[sample_idx].drop(labels=['class', 'Category'], errors='ignore').to_dict()
            samples[f"Class_{cls}"] = sample_row
            if len(samples) >= 2:
                break
                
    return samples

if __name__ == "__main__":
    results = {}
    for ds in ["diabetes", "hcv", "dermatology"]:
        try:
            results[ds] = get_samples(ds)
        except Exception as e:
            results[ds] = {"error": str(e)}
            
    print(json.dumps(results, indent=2))
