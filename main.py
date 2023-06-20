import networkx as nx
import numpy as np

import torch
import torch.nn as nn

from torch_geometric.nn import GINConv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add

from utils import create_dataset


# Hyperparameters
pos_enc_dim = 10
hidden_dim = 64
n_layers = 3

# Construct dataset
Gs = [nx.cycle_graph(50)]*1000
data = create_dataset(Gs, pos_enc_dim=pos_enc_dim)
# TO ADD: write to pickle file here

# Slit into training, validation and test sets
idx = np.random.permutation(len(data))
train_dataset = [data[idx[i]] for i in range(int(0.8*len(data)))]
val_dataset = [data[idx[i]] for i in range(int(0.8*len(data)), int(0.9*len(data)))]
test_dataset = [data[idx[i]] for i in range(int(0.9*len(data)), len(data))]

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=256)
val_loader = DataLoader(val_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Encoder, self).__init__()
        self.n_layers = n_layers

        mp_layers = [GINConv(
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))]

        for i in range(n_layers-1):
            mp_layers.append(GINConv(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                           nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))

        self.mp = nn.ModuleList(mp_layers)

    def forward(self, data):
        x = self.mp[0](data.x, data.edge_index)
        for i in range(1,self.n_layers):
            x = self.mp[i](x, data.edge_index)
        
        x = scatter_add(x, data.batch, dim=0)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Encoder(pos_enc_dim, hidden_dim, n_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for data in train_loader:
    data = data.to(device)
    z = model(data)