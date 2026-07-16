import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import numpy as np

def build_knn_graph(X, k=5):
    """
    Builds a k-Nearest Neighbors graph from the feature matrix X.
    Returns edge_index in PyTorch Geometric format [2, num_edges].
    """
    # Using cosine distance or euclidean. Euclidean is standard for normalized features.
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean') # k+1 because it includes itself
    nn.fit(X)
    
    distances, indices = nn.kneighbors(X)
    
    source_nodes = []
    target_nodes = []
    
    num_nodes = X.shape[0]
    for i in range(num_nodes):
        for j in range(1, k+1): # skip index 0 which is the node itself
            neighbor = indices[i, j]
            source_nodes.append(i)
            target_nodes.append(neighbor)
            
            # Since GAT usually works better with undirected or symmetric graphs,
            # we'll add both directions. PyTorch geometric can also handle directed.
            # Let's just make it undirected by adding both directions.
            source_nodes.append(neighbor)
            target_nodes.append(i)
            
    # Remove duplicates and self loops
    edges = np.array([source_nodes, target_nodes])
    edges = np.unique(edges, axis=1)
    
    # Convert to tensor
    edge_index = torch.tensor(edges, dtype=torch.long)
    return edge_index

def create_pyg_data(X, y, k=5):
    """
    Converts features, labels and k-NN graph into PyTorch Geometric Data object.
    Also creates train and test masks for transductive learning.
    """
    edge_index = build_knn_graph(X, k=k)
    
    x_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    
    # Create train and test masks (80% train, 20% test)
    num_nodes = X.shape[0]
    indices = np.random.permutation(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    
    return data

if __name__ == "__main__":
    from preprocess import get_preprocessed_data
    X, y, _, _ = get_preprocessed_data()
    data = create_pyg_data(X, y, k=5)
    print(f"Data object: {data}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
