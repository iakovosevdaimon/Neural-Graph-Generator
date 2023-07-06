import math
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
from torch import _VF
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add

from softsort import SoftSort
from utils import create_dataset, linear_beta_schedule


# Hyperparameters
pos_enc_dim = 10
epochs = 200
lr = 0.001
hidden_dim = 32
n_layers = 2
n_nodes = 50

# Construct dataset
Gs = [nx.ladder_graph(i) for i in range(10, 26)] + [nx.wheel_graph(i) for i in range(20, 51)] + [nx.cycle_graph(i) for i in range(20, 51)]+[nx.path_graph(i) for i in range(20, 51)]+[nx.star_graph(i) for i in range(19, 50)]
data = create_dataset(Gs, pos_enc_dim, n_nodes)
# TO ADD: write to pickle file here

# Slit into training, validation and test sets
idx = np.random.permutation(len(data))
train_dataset = [data[idx[i]] for i in range(int(0.8*len(data)))]
val_dataset = [data[idx[i]] for i in range(int(0.8*len(data)), int(0.9*len(data)))]
test_dataset = [data[idx[i]] for i in range(int(0.9*len(data)), len(data))]

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

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


# Autoencoder
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

    def encode(self, data):
        x_g, _ = self.encoder(data.x, data.edge_index, data.batch)
        return x_g

    def decode(self, x_g):
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
autoencoder = AutoEncoder(pos_enc_dim, hidden_dim, n_layers, n_nodes).to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

# Train autoencoder
for epoch in range(1, epochs+1):
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = autoencoder.loss_function(data)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    if epoch % 5 == 0:
        print('Epoch: {:03d}, Train Loss: {:.7f}'.format(
                epoch, loss_all))

#################################################################################

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# Loss function for denoising
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# Denoise model
class DenoiseNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(DenoiseNN, self).__init__()
        self.n_layers = n_layers

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        mlp_layers = [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.ModuleList(mlp_layers)

        bn_layers = [nn.BatchNorm1d(hidden_dim) for i in range(n_layers-1)]
        self.bn = nn.ModuleList(bn_layers)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        t = self.time_mlp(t)
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))+t
            x = self.bn[i](x)
        
        x = self.mlp[self.n_layers-1](x)
        return x

timesteps = 300

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

epochs = 5000

denoise_model = DenoiseNN(input_dim=hidden_dim, hidden_dim=256, n_layers=3).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=0.001)

# Train denoising model
for epoch in range(epochs):
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        batch_size = data.num_graphs
        data = data.to(device)

        x_g = autoencoder.encode(data)

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        loss = p_losses(denoise_model, x_g, t, loss_type="huber")
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    if epoch % 100 == 0:
        print('Epoch: {:03d}, Train Loss: {:.7f}'.format(
                epoch, loss_all))
