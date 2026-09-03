import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import numpy as np
import argparse
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from graph_builder import create_pyg_data
from model import HybridClassifier
from train import get_data_for_dataset

def fed_avg(client_models_state, client_sample_counts):
    """
    Perform Federated Averaging (FedAvg).
    client_models_state: list of state_dicts from clients
    client_sample_counts: list of the number of training samples for each client
    """
    total_samples = sum(client_sample_counts)
    global_dict = copy.deepcopy(client_models_state[0])
    
    # Initialize all parameters in global_dict to 0
    for k in global_dict.keys():
        global_dict[k] = torch.zeros_like(global_dict[k], dtype=torch.float)
        
    for i, state in enumerate(client_models_state):
        weight = client_sample_counts[i] / total_samples
        for k in global_dict.keys():
            # Add weighted client weights to global dict
            if state[k].dtype == torch.long:
                # for long tensors like num_batches_tracked, don't average them, just keep the first one
                global_dict[k] = client_models_state[0][k]
            else:
                global_dict[k] += state[k] * weight
                
    return global_dict

def train_local_client(model, data, epochs=5, lr=0.005):
    """
    Train a model locally on a client's data.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
    return copy.deepcopy(model.state_dict())

def evaluate_global_model(model, client_data_list):
    """
    Evaluate the global model on the test sets of all clients.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in client_data_list:
            if data.test_mask.sum() == 0:
                continue
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.test_mask], data.y[data.test_mask])
            total_loss += loss.item()
            
            preds = out.argmax(dim=1)
            all_preds.extend(preds[data.test_mask].cpu().numpy())
            all_labels.extend(data.y[data.test_mask].cpu().numpy())
            
    avg_loss = total_loss / len(client_data_list)
    
    if len(all_labels) == 0:
        return avg_loss, 0, 0, 0, 0
        
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, acc, prec, rec, f1

def run_federated_learning(dataset_name='diabetes', num_clients=5, rounds=50, local_epochs=5):
    # Set seeds for reproducible masks
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n--- Running Federated Learning on {dataset_name.upper()} Dataset ---")
    
    # 1. Load and partition data
    print(f"Loading data and partitioning among {num_clients} clients...")
    X, y, scaler, feature_names = get_data_for_dataset(dataset_name)
    
    in_channels = X.shape[1]
    num_classes = len(np.unique(y))
    print(f"Detected {num_classes} classes and {in_channels} features.")
    
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
    client_data_list = []
    client_sample_counts = []
    
    for fold, (_, client_idx) in enumerate(skf.split(X, y)):
        client_X = X[client_idx]
        client_y = y[client_idx]
        
        # Build local PyG Data object (creates local knn graph and 80/20 train/test split)
        data = create_pyg_data(client_X, client_y, k=5)
        client_data_list.append(data)
        client_sample_counts.append(int(data.train_mask.sum()))
        print(f"  Client {fold+1}: {len(client_y)} total samples, {int(data.train_mask.sum())} train samples")

    # 2. Global Model Initialization
    global_model = HybridClassifier(in_channels=in_channels, num_classes=num_classes, mode='gat')
    
    # 3. Federated Training Loop
    metrics_history = []
    best_f1 = 0.0
    
    os.makedirs(f"federated_models/{dataset_name}", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num}/{rounds} ---")
        
        global_state = copy.deepcopy(global_model.state_dict())
        
        client_states = []
        round_training_time = 0.0
        
        for i in range(num_clients):
            # Client simulation
            local_model = HybridClassifier(in_channels=in_channels, num_classes=num_classes, mode='gat')
            local_model.load_state_dict(global_state) # Receive global parameters
            
            start_time = time.time()
            updated_state = train_local_client(local_model, client_data_list[i], epochs=local_epochs)
            end_time = time.time()
            
            client_states.append(updated_state)
            round_training_time += (end_time - start_time)
            
        # FedAvg Aggregation on Server
        start_agg_time = time.time()
        new_global_state = fed_avg(client_states, client_sample_counts)
        global_model.load_state_dict(new_global_state)
        end_agg_time = time.time()
        
        # Global Evaluation
        avg_loss, acc, prec, rec, f1 = evaluate_global_model(global_model, client_data_list)
        
        print(f"Global Eval - Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
        # Track metrics
        metrics = {
            'Round': round_num,
            'Global_Test_Loss': avg_loss,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'Total_Client_Training_Time_sec': round_training_time,
            'Server_Aggregation_Time_sec': end_agg_time - start_agg_time,
            'Num_Clients': num_clients
        }
        metrics_history.append(metrics)
        
        # Checkpointing
        if round_num % 10 == 0:
            torch.save(global_model.state_dict(), f"federated_models/{dataset_name}/round_{round_num}.pt")
            
        if f1 > best_f1:
            best_f1 = f1
            torch.save(global_model.state_dict(), f"federated_models/{dataset_name}/best_global_model.pt")
            print("  New best model saved!")

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_history)
    csv_path = f"results/{dataset_name}_federated.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nFederated training complete. Metrics saved to {csv_path}")
    
    return global_model, metrics_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning for GAT Models')
    parser.add_argument('--dataset', type=str, default='diabetes', choices=['diabetes', 'hcv', 'dermatology'], help='Dataset to train on')
    parser.add_argument('--clients', type=int, default=5, help='Number of federated clients')
    parser.add_argument('--rounds', type=int, default=50, help='Number of communication rounds')
    parser.add_argument('--epochs', type=int, default=5, help='Number of local epochs per round')
    
    args = parser.parse_args()
    
    run_federated_learning(dataset_name=args.dataset, num_clients=args.clients, rounds=args.rounds, local_epochs=args.epochs)
