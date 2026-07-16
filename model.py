import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT_Module(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=16, heads=1):
        super(GAT_Module, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index, return_attention_weights=False):
        if return_attention_weights:
            x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
            x = F.relu(x)
            return x, (att1, att2)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            return x

class RBMLayer(torch.nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super(RBMLayer, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.1)
        self.h_bias = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.v_bias = torch.nn.Parameter(torch.zeros(visible_dim))
    
    def sample_h(self, v):
        h_prob = torch.sigmoid(F.linear(v, self.W.t(), self.h_bias))
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        v_prob = torch.sigmoid(F.linear(h, self.W, self.v_bias))
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, v):
        # Forward pass for fine-tuning
        return torch.sigmoid(F.linear(v, self.W.t(), self.h_bias))

    def contrastive_divergence(self, v, lr=0.01):
        # CD-1 step
        h0_prob, h0_sample = self.sample_h(v)
        v1_prob, v1_sample = self.sample_v(h0_sample)
        h1_prob, _ = self.sample_h(v1_sample)

        positive_grad = torch.matmul(v.t(), h0_prob)
        negative_grad = torch.matmul(v1_sample.t(), h1_prob)
        
        self.W.data += lr * (positive_grad - negative_grad) / v.size(0)
        self.v_bias.data += lr * torch.mean(v - v1_sample, dim=0)
        self.h_bias.data += lr * torch.mean(h0_prob - h1_prob, dim=0)
        
        return F.mse_loss(v, v1_prob)

class RDBN_Module(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[16, 16], num_classes=2):
        super(RDBN_Module, self).__init__()
        self.layers = torch.nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            self.layers.append(RBMLayer(dims[i], dims[i+1]))
        self.classifier = torch.nn.Linear(hidden_dims[-1], num_classes)
        
    def forward(self, x, return_contributions=False):
        h = x
        h_prev = x
        contributions = []
        for i, layer in enumerate(self.layers):
            h_new = layer(h)
            # Add residual connection
            if i > 0 and h_new.shape == h_prev.shape:
                h_new = h_new + h_prev
                if return_contributions:
                    contributions.append({
                        'layer': i,
                        'residual_norm': float(torch.norm(h_prev).item()),
                        'new_norm': float(torch.norm(h_new - h_prev).item())
                    })
            h_prev = h
            h = h_new
        
        out = self.classifier(h)
        if return_contributions:
            return out, contributions
        return out

class HybridClassifier(torch.nn.Module):
    def __init__(self, in_channels, num_classes=2, mode='hybrid'):
        """
        mode: 'gat', 'rdbn', or 'hybrid'
        """
        super(HybridClassifier, self).__init__()
        self.mode = mode
        self.gat = GAT_Module(in_channels, out_channels=16)
        
        rdbn_in = in_channels if mode == 'rdbn' else 16
        self.rdbn = RDBN_Module(input_dim=rdbn_in, hidden_dims=[16, 16], num_classes=num_classes)
        
        if mode == 'gat':
            self.gat_classifier = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index=None, return_explainability=False):
        att = None
        contributions = None
        if self.mode == 'gat':
            if return_explainability:
                emb, att = self.gat(x, edge_index, return_attention_weights=True)
            else:
                emb = self.gat(x, edge_index)
            out = self.gat_classifier(emb)
            if return_explainability:
                return out, {'attention': att, 'rdbn_contributions': None}
            return out
        
        elif self.mode == 'rdbn':
            if return_explainability:
                out, contributions = self.rdbn(x, return_contributions=True)
            else:
                out = self.rdbn(x)
            if return_explainability:
                return out, {'attention': None, 'rdbn_contributions': contributions}
            return out
            
        elif self.mode == 'hybrid':
            if return_explainability:
                emb, att = self.gat(x, edge_index, return_attention_weights=True)
                out, contributions = self.rdbn(emb, return_contributions=True)
                return out, {'attention': att, 'rdbn_contributions': contributions}
            else:
                emb = self.gat(x, edge_index)
                out = self.rdbn(emb)
                return out

if __name__ == "__main__":
    model = HybridClassifier(in_channels=16, mode='hybrid')
    print(model)

