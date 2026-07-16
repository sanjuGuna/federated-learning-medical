import torch
import torch.nn as nn
import torch.optim as optim
import os
from preprocess import get_preprocessed_data
from graph_builder import create_pyg_data
from model import GATClassifier

def train_model(epochs=100, save_path="saved_models/gat_model.pth"):
    # Set seeds for reproducible masks
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)

    # 1. Prepare Data
    print("Loading and preprocessing data...")
    X, y, scaler, feature_names = get_preprocessed_data()
    
    print("Building k-NN graph and converting to PyTorch Geometric data...")
    data = create_pyg_data(X, y, k=5)
    
    in_channels = X.shape[1] # should be 16
    print(f"Number of features (in_channels): {in_channels}")
    
    # 2. Initialize Model, Loss, Optimizer
    model = GATClassifier(in_channels=in_channels)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # 3. Training Loop
    model.train()
    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Calculate loss on training nodes only
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate training accuracy
        pred = out.argmax(dim=1)
        train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        train_acc = int(train_correct) / int(data.train_mask.sum())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}/{epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}')
            
    # 4. Save Model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
if __name__ == "__main__":
    train_model(epochs=100)
