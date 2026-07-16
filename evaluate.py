import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocess import get_preprocessed_data
from graph_builder import create_pyg_data
from model import HybridClassifier

def evaluate_models():
    print("Loading data...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    X, y, scaler, feature_names = get_preprocessed_data()
    data = create_pyg_data(X, y, k=5)
    
    in_channels = X.shape[1]
    
    modes = ['gat', 'rdbn', 'hybrid']
    paths = {
        'gat': 'saved_models/gat_only.pth',
        'rdbn': 'saved_models/rdbn_only.pth',
        'hybrid': 'saved_models/hybrid.pth'
    }
    
    results = []
    
    for mode in modes:
        model_path = paths[mode]
        print(f"\n--- Evaluating {mode.upper()} ---")
        try:
            model = HybridClassifier(in_channels=in_channels, mode=mode)
            model.load_state_dict(torch.load(model_path))
            model.eval()
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Skipping...")
            continue
            
        with torch.no_grad():
            if mode in ['gat', 'hybrid']:
                out, explain = model(data.x, data.edge_index, return_explainability=True)
            else:
                out, explain = model(data.x, return_explainability=True)
                
            probs = torch.softmax(out, dim=1)[:, 1].numpy()
            preds = out.argmax(dim=1).numpy()
            
            y_true = data.y.numpy()
            test_mask = data.test_mask.numpy()
            
            y_true_test = y_true[test_mask]
            preds_test = preds[test_mask]
            probs_test = probs[test_mask]
            
            acc = accuracy_score(y_true_test, preds_test)
            prec = precision_score(y_true_test, preds_test, zero_division=0)
            rec = recall_score(y_true_test, preds_test, zero_division=0)
            f1 = f1_score(y_true_test, preds_test, zero_division=0)
            roc_auc = roc_auc_score(y_true_test, probs_test)
            
            results.append({
                'Model': mode.upper(),
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            })
            
            # Print Explainability for Hybrid
            if mode == 'hybrid':
                print("\n[Explainability Analysis - Hybrid]")
                # 1. Attention (just check shapes)
                att = explain.get('attention')
                if att:
                    print("GAT Attention extracted successfully.")
                    # att is tuple (att1, att2). att1 is (edge_index, edge_attr)
                    edges, weights = att[0]
                    print(f"Layer 1 Max Attention Weight: {weights.max().item():.4f}")
                
                # 2. RDBN Contributions
                contribs = explain.get('rdbn_contributions')
                if contribs:
                    print("RDBN Residual Path Contributions:")
                    for c in contribs:
                        print(f"  Layer {c['layer']}: Residual Norm = {c['residual_norm']:.4f}, New Transform Norm = {c['new_norm']:.4f}")
                    # Find which path contributed most
                    max_residual_layer = max(contribs, key=lambda x: x['residual_norm'])['layer']
                    print(f"-> Residual connection at Layer {max_residual_layer} contributed the most information.")

    if results:
        print("\n=== ABLATION STUDY RESULTS ===")
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))

if __name__ == "__main__":
    evaluate_models()
