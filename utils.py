import networkx as nx
import numpy as np
import torch

from scipy import sparse as sp
from torch import Tensor
from torch_geometric.data import Data


def positional_encoding(row, col, n, pos_enc_dim=10):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n), dtype=float)
    degs = A.dot(np.ones(n))
    #N = sp.diags(np.power(degs, -0.5))
    #L = sp.eye(n) - N * A * N
    L = sp.diags(degs) - A

    # Eigenvectors with numpy
    x = np.zeros((n,pos_enc_dim))
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    x[:,:min(n-1, pos_enc_dim)] = EigVec[:,1:pos_enc_dim+1]

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # x[:,:min(n-1, pos_enc_dim)] = EigVec[:,1:pos_enc_dim+1]
    
    return x


def create_dataset(Gs, pos_enc_dim, max_n_nodes):
    data = []
    for G in Gs:
        n = G.number_of_nodes()
        row, col = [], []
        for edge in G.edges():
            row.append(edge[0])
            col.append(edge[1])
            
            row.append(edge[1])
            col.append(edge[0])

        x = positional_encoding(row, col, n, pos_enc_dim)
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        adj = torch.zeros(max_n_nodes, max_n_nodes)
        adj[edge_index[0,:], edge_index[1,:]] = 1
        data.append(Data(x=x, edge_index=edge_index, adj=adj))

    return data


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
