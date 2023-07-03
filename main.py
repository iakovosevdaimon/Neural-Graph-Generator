import networkx as nx
import numpy as np

import torch
import torch.nn as nn
from torch import _VF
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence,PackedSequence,pad_packed_sequence
from torch_geometric.nn import GINConv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add

from softsort import SoftSort
from utils import create_dataset


# Hyperparameters
pos_enc_dim = 10
epochs = 200
lr = 0.001
hidden_dim = 32
n_layers = 2
n_nodes = 50

# Construct dataset
Gs = [nx.cycle_graph(i) for i in range(20, 51)]+[nx.path_graph(i) for i in range(20, 51)]+[nx.star_graph(i) for i in range(19, 50)]
data = create_dataset(Gs, pos_enc_dim, n_nodes)
# TO ADD: write to pickle file here

# Slit into training, validation and test sets
idx = np.random.permutation(len(data))
train_dataset = [data[idx[i]] for i in range(int(0.8*len(data)))]
val_dataset = [data[idx[i]] for i in range(int(0.8*len(data)), int(0.9*len(data)))]
test_dataset = [data[idx[i]] for i in range(int(0.9*len(data)), len(data))]

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

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

    def forward(self, x_in, edge_index, batch):
        x = self.mp[0](x_in, edge_index)
        for i in range(1,self.n_layers):
            x = self.mp[i](x, edge_index)

        x_g = scatter_add(x, batch, dim=0)
        return x_g, x

# Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-1)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


# Decoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_nodes):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.n_nodes = n_nodes
        self.encoder = Encoder(input_dim, hidden_dim, n_layers)
        self.decoder = Decoder(hidden_dim, n_layers, n_nodes)
        self.fc = nn.Linear(hidden_dim, 1)
        self.softsort = SoftSort()

    def forward(self, data):
        x_g, _ = self.encoder(data.x, data.edge_index, data.batch)
        adj = self.decoder(x_g)
        return adj

    def loss_function(self, data):
        x_g, x  = self.encoder(data.x, data.edge_index, data.batch)
        adj = self.decoder(x_g)

        with torch.no_grad():
            degs = torch.bmm(adj, torch.ones(adj.size(0), adj.size(1), 1, device=x.device)).squeeze()
            #degs = torch.pow(degs, -0.5)
            #N = torch.diag_embed(degs)
            #L = torch.eye(adj.size(1), device=x.device) - torch.bmm(torch.bmm(N, adj), N)
            L = torch.diag_embed(degs) - adj
            _, EigVec = torch.linalg.eig(L)
            EigVec = torch.real(EigVec)
            h = EigVec[:,:,1:self.input_dim+1]
            h = torch.reshape(h, (h.size(0)*h.size(1), h.size(2)))
            
            adj_reshaped = torch.reshape(adj, (adj.size(0)*adj.size(1), adj.size(2)))
            idx = torch.nonzero(adj_reshaped)
            batch = torch.reshape(torch.arange(adj.size(0), device=x.device).repeat(adj.size(1), 1).T, (-1, 1)).squeeze()
            _, x_gen = self.encoder(h, idx.T, batch)
        scores = self.fc(x_gen).squeeze()
        scores = torch.reshape(scores, (adj.size(0), adj.size(1)))
        P = self.softsort(scores)
        
        adj_perm = torch.einsum("abc,acd->abd", (P, adj))
        adj_perm = torch.einsum("abc,adc->abd", (adj_perm, P))
            
        scores = self.fc(x).squeeze()
        with torch.no_grad():
            _, lens = torch.unique(data.batch, return_counts=True)
            idx = torch.cat([torch.arange(lens[i], device=x.device)+(i*self.n_nodes) for i in range(lens.size(0))], dim=0)
            scores = torch.zeros(self.n_nodes*x_g.size(0), device=x.device).scatter_(0, idx, scores)
            scores = torch.reshape(scores, (x_g.size(0), -1))
        P = self.softsort(scores)
        adj_in = torch.reshape(data.adj, (-1, data.adj.size(1), data.adj.size(1)))
        adj_perm_in = torch.einsum("abc,acd->abd", (P, adj_in))
        adj_perm_in = torch.einsum("abc,adc->abd", (adj_perm_in, P))

        loss = F.l1_loss(adj_perm, adj_perm_in)
        return loss
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoEncoder(pos_enc_dim, hidden_dim, n_layers, n_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)



best_val_loss, test_acc = 100, 0
for epoch in range(1, epochs+1):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = model.loss_function(data)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    if epoch % 5 == 0:
        print('Epoch: {:03d}, Train Loss: {:.7f}'.format(
                epoch, loss_all))
