import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=2, heads=1):
        super(GATClassifier, self).__init__()
        # First GAT layer
        # Output dim will be hidden_channels * heads
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        
        # Second GAT layer
        # Input dim needs to match previous output dim
        # We'll use 1 head for the second layer for simplicity
        # Wait, the prompt says GATConv(32 -> 16).
        self.conv2 = GATConv(hidden_channels * heads, 16, heads=1)
        
        # Linear output layer
        self.linear = torch.nn.Linear(16, out_channels)

    def forward(self, x, edge_index):
        # Layer 1: GATConv -> ReLU -> Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Layer 2: GATConv -> ReLU (optional, typically used before linear)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Output Layer: Linear
        x = self.linear(x)
        
        return x

if __name__ == "__main__":
    # Test model definition
    model = GATClassifier(in_channels=16) # 16 features in diabetes dataset
    print(model)
