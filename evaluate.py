import torch
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from graph_builder import create_pyg_data
from model import HybridClassifier

def get_data_for_dataset(dataset_name):
    if dataset_name == 'diabetes':
        from preprocess import get_preprocessed_data
        return get_preprocessed_data()
    elif dataset_name == 'hcv':
        from hcv_preprocess import get_preprocessed_data
        return get_preprocessed_data()
    elif dataset_name == 'dermatology':
        from dermatology_preprocess import get_preprocessed_data
        return get_preprocessed_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_evaluation_metrics(dataset_name='diabetes'):
    torch.manual_seed(42)
    np.random.seed(42)
    
    X, y, scaler, feature_names = get_data_for_dataset(dataset_name)
    data = create_pyg_data(X, y, k=5)
    
    in_channels = X.shape[1]
    num_classes = len(np.unique(y))
    
    modes = ['gat', 'rdbn', 'hybrid']
    
    results = []
    hybrid_cm = None
    
    for mode in modes:
        model_path = f"saved_models/{dataset_name}/{mode}.pth"
        if not os.path.exists(model_path):
            # Fallback for diabetes old structure
            if dataset_name == 'diabetes':
                alt_paths = {'gat': 'saved_models/gat_only.pth', 'rdbn': 'saved_models/rdbn_only.pth', 'hybrid': 'saved_models/hybrid.pth'}
                model_path = alt_paths[mode]
                if not os.path.exists(model_path):
                    continue
            else:
                continue
                
        model = HybridClassifier(in_channels=in_channels, num_classes=num_classes, mode=mode)
        model.load_state_dict(torch.load(model_path))
        model.eval()
            
        with torch.no_grad():
            if mode in ['gat', 'hybrid']:
                out, explain = model(data.x, data.edge_index, return_explainability=True)
            else:
                out, explain = model(data.x, return_explainability=True)
                
            probs = torch.softmax(out, dim=1).numpy()
            preds = out.argmax(dim=1).numpy()
            
            y_true = data.y.numpy()
            test_mask = data.test_mask.numpy()
            
            y_true_test = y_true[test_mask]
            preds_test = preds[test_mask]
            probs_test = probs[test_mask]
            
            acc = accuracy_score(y_true_test, preds_test)
            
            if num_classes > 2:
                prec = precision_score(y_true_test, preds_test, zero_division=0, average='macro')
                rec = recall_score(y_true_test, preds_test, zero_division=0, average='macro')
                f1 = f1_score(y_true_test, preds_test, zero_division=0, average='macro')
                try:
                    roc_auc = roc_auc_score(y_true_test, probs_test, multi_class='ovr')
                except:
                    roc_auc = 0.0
            else:
                prec = precision_score(y_true_test, preds_test, zero_division=0)
                rec = recall_score(y_true_test, preds_test, zero_division=0)
                f1 = f1_score(y_true_test, preds_test, zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_true_test, probs_test[:, 1])
                except:
                    roc_auc = 0.0
            
            if mode == 'hybrid':
                hybrid_cm = confusion_matrix(y_true_test, preds_test).tolist()
                
            results.append({
                'Dataset': dataset_name.upper(),
                'Model': mode.upper(),
                'Accuracy': round(acc * 100, 1),
                'Precision': round(prec * 100, 1),
                'Recall': round(rec * 100, 1),
                'F1-Score': round(f1 * 100, 1),
                'ROC-AUC': round(roc_auc, 2)
            })

    return {
        "ablation": results,
        "confusion_matrix": hybrid_cm
    }

def evaluate_models(dataset_name='diabetes'):
    metrics = get_evaluation_metrics(dataset_name)
    
    if metrics["ablation"]:
        print(f"\n=== {dataset_name.upper()} ABLATION STUDY RESULTS ===")
        df_results = pd.DataFrame(metrics["ablation"])
        print(df_results.to_string(index=False))
        
        # Save to csv for visualization script
        os.makedirs("results", exist_ok=True)
        df_results.to_csv(f"results/{dataset_name}_results.csv", index=False)
        
        # Save confusion matrix
        if metrics["confusion_matrix"]:
            cm_df = pd.DataFrame(metrics["confusion_matrix"])
            cm_df.to_csv(f"results/{dataset_name}_hybrid_cm.csv", index=False)
    else:
        print(f"No trained models found for dataset: {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Models')
    parser.add_argument('--dataset', type=str, default='diabetes', choices=['diabetes', 'hcv', 'dermatology'], help='Dataset to evaluate')
    args = parser.parse_args()
    
    evaluate_models(args.dataset)
