import torch
import numpy as np
import pandas as pd
from preprocess import load_data, encode_features, normalize_age, get_preprocessed_data
from graph_builder import create_pyg_data
from model import GATClassifier

def predict_new_patient(patient_data_dict, model_path="saved_models/gat_model.pth", filepath="archive/diabetes_data_upload.csv"):
    """
    Predicts diabetes for a new patient by dynamically injecting them into the graph.
    """
    # 1. Load original data to rebuild graph
    torch.manual_seed(42)
    np.random.seed(42)
    
    # We will get original preprocessed data just to append to it
    # For a real system, we'd save the `scaler` and original `X` instead of reprocessing everything.
    # But for Phase 1 simplicity, we just re-run the pipeline.
    df_orig = load_data(filepath)
    df_orig = encode_features(df_orig)
    df_orig, scaler = normalize_age(df_orig)
    
    y_orig = df_orig["class"].values
    X_orig = df_orig.drop(columns=["class"]).values
    
    # 2. Preprocess new patient
    # Convert dict to DataFrame row
    df_new = pd.DataFrame([patient_data_dict])
    
    # Ensure it has the same columns (except class)
    if "class" in df_new.columns:
        df_new = df_new.drop(columns=["class"])
        
    df_new = encode_features(df_new)
    df_new, _ = normalize_age(df_new, scaler=scaler)
    
    # We assume feature order matches original
    X_new = df_new[df_orig.columns.drop("class")].values
    
    # Target is unknown, we can just append a dummy 0
    y_new = np.array([0])
    
    # 3. Append to existing graph data
    X_combined = np.vstack([X_orig, X_new])
    y_combined = np.concatenate([y_orig, y_new])
    
    # Build a new PyG data object which will compute k-NN including the new node
    data = create_pyg_data(X_combined, y_combined, k=5)
    
    # The new node is the last node in the graph
    new_node_idx = X_combined.shape[0] - 1
    
    # 4. Load Model
    in_channels = X_combined.shape[1]
    model = GATClassifier(in_channels=in_channels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 5. Predict
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)
        
        # Get prediction for the newly added node
        new_patient_prob = probs[new_node_idx].numpy()
        pred_class = np.argmax(new_patient_prob)
        
        # class 1 is Positive, 0 is Negative
        pred_label = "Positive" if pred_class == 1 else "Negative"
        confidence = new_patient_prob[pred_class] * 100
        
        print("\n--- Prediction Results ---")
        print(f"Prediction: {pred_label}")
        print(f"Probability: {confidence:.1f}%")
        
        return pred_label, confidence

if __name__ == "__main__":
    # Example input from the user prompt
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
