import torch
import numpy as np
from model import HybridClassifier
from preprocess import get_preprocessed_data
from graph_builder import create_pyg_data

def test_fl_model():
    X, y, scaler, feature_names = get_preprocessed_data()
    data = create_pyg_data(X, y, k=5)
    
    in_channels = X.shape[1]
    num_classes = len(np.unique(y))
    
    model = HybridClassifier(in_channels=in_channels, num_classes=num_classes, mode='gat')
    model.load_state_dict(torch.load("federated_models/diabetes/best_global_model.pt"))
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)
        
        acc = (preds == data.y).float().mean().item()
        print(f"Federated GAT Model loaded successfully!")
        print(f"Federated GAT Model Overall Accuracy on full Diabetes dataset: {acc * 100:.2f}%")
        
        print("\nChecking HCV Federated Model...")
        from hcv_preprocess import get_preprocessed_data as hcv_data
        X_h, y_h, _, _ = hcv_data()
        data_h = create_pyg_data(X_h, y_h, k=5)
        model_h = HybridClassifier(in_channels=X_h.shape[1], num_classes=len(np.unique(y_h)), mode='gat')
        model_h.load_state_dict(torch.load("federated_models/hcv/best_global_model.pt"))
        model_h.eval()
        out_h = model_h(data_h.x, data_h.edge_index)
        acc_h = (out_h.argmax(dim=1) == data_h.y).float().mean().item()
        print(f"Federated HCV Model Overall Accuracy: {acc_h * 100:.2f}%")

if __name__ == "__main__":
    test_fl_model()
