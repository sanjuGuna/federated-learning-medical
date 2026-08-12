import torch
import numpy as np
import pandas as pd
from graph_builder import create_pyg_data
from model import HybridClassifier
import os

# HCV Categories (sorted alphabetically as LabelEncoder would)
HCV_CLASSES = ['0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis']

# Dermatology Classes
DERMATOLOGY_CLASSES = [
    'Psoriasis', 
    'Seborrheic dermatitis', 
    'Lichen planus', 
    'Pityriasis rosea', 
    'Chronic dermatitis', 
    'Pityriasis rubra pilaris'
]

def predict_patient(patient_data_dict, dataset_name="diabetes"):
    """
    Predicts for a new patient dynamically depending on the dataset.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load original data and preprocess
    if dataset_name == "diabetes":
        from preprocess import load_data, encode_features, normalize_age, get_preprocessed_data
        filepath = "archive/diabetes_data_upload.csv"
        df_orig = load_data(filepath)
        df_orig = encode_features(df_orig)
        df_orig, scaler = normalize_age(df_orig)
        
        y_orig = df_orig["class"].values
        X_orig = df_orig.drop(columns=["class"]).values
        feature_names = df_orig.columns.drop("class").tolist()
        
        # Preprocess new patient
        df_new = pd.DataFrame([patient_data_dict])
        df_new = encode_features(df_new)
        df_new, _ = normalize_age(df_new, scaler=scaler)
        X_new = df_new[feature_names].values
        
    elif dataset_name == "hcv":
        from hcv_preprocess import load_data, preprocess_hcv, normalize_features
        filepath = "archive/hcvdat0.csv"
        df_orig = load_data(filepath)
        df_orig, le = preprocess_hcv(df_orig)
        df_orig, scaler = normalize_features(df_orig)
        
        y_orig = df_orig["Category"].values
        X_orig = df_orig.drop(columns=["Category"]).values
        feature_names = df_orig.columns.drop("Category").tolist()
        
        # Preprocess new patient
        df_new = pd.DataFrame([patient_data_dict])
        
        # Encode sex
        if "Sex" in df_new.columns:
            df_new["Sex"] = df_new["Sex"].map({"m": 0, "f": 1})
            
        df_new, _ = normalize_features(df_new, scaler=scaler)
        X_new = df_new[feature_names].values
        
    elif dataset_name == "dermatology":
        from dermatology_preprocess import load_data, preprocess_dermatology, normalize_features
        filepath = "archive/dermatology.data"
        df_orig = load_data(filepath)
        df_orig = preprocess_dermatology(df_orig)
        df_orig, scaler = normalize_features(df_orig)
        
        y_orig = df_orig["class"].values
        X_orig = df_orig.drop(columns=["class"]).values
        feature_names = df_orig.columns.drop("class").tolist()
        
        # Preprocess new patient
        df_new = pd.DataFrame([patient_data_dict])
        for col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
        df_new, _ = normalize_features(df_new, scaler=scaler)
        X_new = df_new[feature_names].values
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Dummy target for new patient
    y_new = np.array([0])
    
    # 3. Append to existing graph data
    X_combined = np.vstack([X_orig, X_new])
    y_combined = np.concatenate([y_orig, y_new])
    
    data = create_pyg_data(X_combined, y_combined, k=5)
    new_node_idx = X_combined.shape[0] - 1
    
    # Extract neighbors
    edges = data.edge_index.numpy()
    neighbors = edges[1, edges[0] == new_node_idx].tolist()
    
    # 4. Load Model
    in_channels = X_combined.shape[1]
    num_classes = len(np.unique(y_orig))
    
    model_path = f"saved_models/{dataset_name}/hybrid.pth"
    if not os.path.exists(model_path) and dataset_name == 'diabetes':
        model_path = "saved_models/hybrid.pth"
        
    model = HybridClassifier(in_channels=in_channels, num_classes=num_classes, mode='hybrid')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 5. Predict
    with torch.no_grad():
        out, explain = model(data.x, data.edge_index, return_explainability=True)
        probs = torch.softmax(out, dim=1)
        
        new_patient_prob = probs[new_node_idx].numpy()
        pred_class = np.argmax(new_patient_prob)
        
        if dataset_name == "diabetes":
            pred_label = "Positive" if pred_class == 1 else "Negative"
        elif dataset_name == "hcv":
            pred_label = HCV_CLASSES[pred_class]
        elif dataset_name == "dermatology":
            pred_label = DERMATOLOGY_CLASSES[pred_class]
            
        confidence = float(new_patient_prob[pred_class] * 100)
        
        # Process explainability
        neighbor_importance = {}
        feature_importance = {}
        att = explain.get('attention')
        if att:
            edge_idx, edge_weights = att[0]
            edge_idx = edge_idx.numpy()
            edge_weights = edge_weights.numpy()
            
            mask = edge_idx[0] == new_node_idx
            dest_nodes = edge_idx[1, mask]
            weights = edge_weights[mask]
            
            feature_imp_scores = np.zeros(len(feature_names))
            
            for dest, w in zip(dest_nodes, weights):
                neighbor_importance[f"Neighbor_{dest}"] = float(w[0])
                for f_idx in range(len(feature_names)):
                    diff = abs(X_new[0, f_idx] - X_combined[dest, f_idx])
                    similarity = 1.0 - diff
                    feature_imp_scores[f_idx] += w[0] * similarity
                    
            if len(dest_nodes) > 0:
                feature_imp_scores = feature_imp_scores / np.sum(weights)
                
            for f_idx, name in enumerate(feature_names):
                feature_importance[name] = float(feature_imp_scores[f_idx])
                
        layer_contributions = []
        contribs = explain.get('rdbn_contributions')
        if contribs:
            for c in contribs:
                layer_contributions.append({
                    "layer": c['layer'],
                    "residual_norm": c['residual_norm'],
                    "new_transform_norm": c['new_norm']
                })
        
        result_dict = {
            "prediction": pred_label,
            "confidence": confidence,
            "neighbor_ids": neighbors,
            "neighbor_importance": neighbor_importance,
            "feature_importance": feature_importance,
            "per_layer_contribution": layer_contributions
        }
        
        return result_dict

# Alias for backward compatibility
def predict_new_patient(patient_data_dict):
    return predict_patient(patient_data_dict, dataset_name="diabetes")
