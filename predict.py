import torch
import numpy as np
import pandas as pd
from preprocess import load_data, encode_features, normalize_age, get_preprocessed_data
from graph_builder import create_pyg_data
from model import HybridClassifier

def predict_new_patient(patient_data_dict, model_path="saved_models/hybrid.pth", filepath="archive/diabetes_data_upload.csv"):
    """
    Predicts diabetes for a new patient by dynamically injecting them into the graph.
    Returns a structured dictionary with predictions and explainability signals.
    """
    # 1. Load original data to rebuild graph
    torch.manual_seed(42)
    np.random.seed(42)
    
    df_orig = load_data(filepath)
    df_orig = encode_features(df_orig)
    df_orig, scaler = normalize_age(df_orig)
    
    y_orig = df_orig["class"].values
    X_orig = df_orig.drop(columns=["class"]).values
    
    # 2. Preprocess new patient
    df_new = pd.DataFrame([patient_data_dict])
    if "class" in df_new.columns:
        df_new = df_new.drop(columns=["class"])
        
    df_new = encode_features(df_new)
    df_new, _ = normalize_age(df_new, scaler=scaler)
    X_new = df_new[df_orig.columns.drop("class")].values
    
    y_new = np.array([0])
    
    # 3. Append to existing graph data
    X_combined = np.vstack([X_orig, X_new])
    y_combined = np.concatenate([y_orig, y_new])
    
    data = create_pyg_data(X_combined, y_combined, k=5)
    new_node_idx = X_combined.shape[0] - 1
    
    # Extract neighbors of the new node from edge_index
    edges = data.edge_index.numpy()
    neighbors = edges[1, edges[0] == new_node_idx].tolist()
    
    # 4. Load Model
    in_channels = X_combined.shape[1]
    model = HybridClassifier(in_channels=in_channels, mode='hybrid')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 5. Predict
    with torch.no_grad():
        out, explain = model(data.x, data.edge_index, return_explainability=True)
        probs = torch.softmax(out, dim=1)
        
        new_patient_prob = probs[new_node_idx].numpy()
        pred_class = np.argmax(new_patient_prob)
        
        pred_label = "Positive" if pred_class == 1 else "Negative"
        confidence = float(new_patient_prob[pred_class] * 100)
        
        # Process explainability
        feature_importance = {}
        att = explain.get('attention')
        if att:
            # We don't have explicit node features importance from GAT directly, 
            # but we can provide the attention weights on its edges
            edge_idx, edge_weights = att[0] # Layer 1 attention
            edge_idx = edge_idx.numpy()
            edge_weights = edge_weights.numpy()
            
            # Find edges where source is new_node_idx
            mask = edge_idx[0] == new_node_idx
            dest_nodes = edge_idx[1, mask]
            weights = edge_weights[mask]
            
            for dest, w in zip(dest_nodes, weights):
                feature_importance[f"Neighbor_{dest}"] = float(w[0])
                
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
            "feature_importance": feature_importance,
            "per_layer_contribution": layer_contributions
        }
        
        print("\n--- Prediction Results ---")
        for k, v in result_dict.items():
            print(f"{k}: {v}")
            
        return result_dict

if __name__ == "__main__":
    example_patient = {
        "Age": 45,
        "Gender": "Male",
        "Polyuria": "Yes",
        "Polydipsia": "No",
        "sudden weight loss": "No",
        "weakness": "Yes",
        "Polyphagia": "No",
        "Genital thrush": "No",
        "visual blurring": "No",
        "Itching": "Yes",
        "Irritability": "No",
        "delayed healing": "Yes",
        "partial paresis": "No",
        "muscle stiffness": "Yes",
        "Alopecia": "Yes",
        "Obesity": "Yes"
    }
    
    predict_new_patient(example_patient)
