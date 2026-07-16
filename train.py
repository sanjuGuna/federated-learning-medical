import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from preprocess import get_preprocessed_data
from graph_builder import create_pyg_data
from model import HybridClassifier

def pretrain_rdbn(model, data, epochs=50, lr=0.01):
    print("Pretraining RDBN layers...")
    model.eval() # Freeze GAT if present
    
    with torch.no_grad():
        if model.mode == 'hybrid':
            # Extract embeddings
            embeddings = model.gat(data.x, data.edge_index)
        else:
            embeddings = data.x
            
    # Iterate over RBMLayers
    current_input = embeddings
    for i, layer in enumerate(model.rdbn.layers):
        print(f"  Pretraining layer {i+1}...")
        for epoch in range(epochs):
            loss = layer.contrastive_divergence(current_input, lr=lr)
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Reconstruction Loss: {loss:.4f}")
        
        with torch.no_grad():
            current_input = layer(current_input)
            
def train_model(mode='hybrid', epochs=100, pretrain_epochs=50, save_path="saved_models/gat_model.pth"):
    # Set seeds for reproducible masks
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n--- Training {mode.upper()} Model ---")
    print("Loading and preprocessing data...")
    X, y, scaler, feature_names = get_preprocessed_data()
    
    print("Building k-NN graph and converting to PyTorch Geometric data...")
    data = create_pyg_data(X, y, k=5)
    
    in_channels = X.shape[1]
    
    model = HybridClassifier(in_channels=in_channels, mode=mode)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # Pretraining Phase
    if mode in ['rdbn', 'hybrid']:
        pretrain_rdbn(model, data, epochs=pretrain_epochs)
        
    # Fine-tuning / Training Phase
    print("Starting joint training/fine-tuning...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if mode in ['gat', 'hybrid']:
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x)
            
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        pred = out.argmax(dim=1)
        train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        train_acc = int(train_correct) / int(data.train_mask.sum())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}/{epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}')
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model('gat', save_path="saved_models/gat_only.pth")
    train_model('rdbn', save_path="saved_models/rdbn_only.pth")
    train_model('hybrid', save_path="saved_models/hybrid.pth")
