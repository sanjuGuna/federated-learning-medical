import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import get_preprocessed_data
from graph_builder import create_pyg_data
from model import GATClassifier

def evaluate_model(model_path="saved_models/gat_model.pth"):
    print("Loading data...")
    # Fix the random seed if you want consistent train/test masks across runs
    # But for a simple script, we'll just evaluate on whatever mask is generated,
    # or ideally we'd save the mask. Since we don't save the mask, we'll evaluate 
    # on the whole graph just to see overall performance, or a random split.
    # To keep it rigorous, we'll set a fixed seed before create_pyg_data so it matches train.py
    # if train.py was run with the same seed. Let's just evaluate on the whole graph for simplicity in this phase,
    # or use the generated test_mask (which will be a random 20% each run).
    torch.manual_seed(42)
    np.random.seed(42)
    
    X, y, scaler, feature_names = get_preprocessed_data()
    data = create_pyg_data(X, y, k=5)
    
    in_channels = X.shape[1]
    
    print("Loading model...")
    model = GATClassifier(in_channels=in_channels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Running evaluation on test mask...")
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)[:, 1].numpy() # probability of class 1
        preds = out.argmax(dim=1).numpy()
        
        y_true = data.y.numpy()
        
        # Evaluate only on test mask
        test_mask = data.test_mask.numpy()
        y_true_test = y_true[test_mask]
        preds_test = preds[test_mask]
        probs_test = probs[test_mask]
        
        acc = accuracy_score(y_true_test, preds_test)
        prec = precision_score(y_true_test, preds_test)
        rec = recall_score(y_true_test, preds_test)
        f1 = f1_score(y_true_test, preds_test)
        roc_auc = roc_auc_score(y_true_test, probs_test)
        cm = confusion_matrix(y_true_test, preds_test)
        
        print("\n--- Evaluation Metrics (Test Set) ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)

if __name__ == "__main__":
    evaluate_model()
