import os
import networkx as nx
import numpy as np
import scipy as sp

import torch
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import Dataset

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


class CustomDataset(Dataset):
    """ Based on https://github.com/lrjconan/GRAN/blob/master/utils/data_helper.py#L192 """

    def __init__(self, k, same_sample=False, ignore_first_eigv=False):
        min_num_nodes=20
        max_num_nodes=50
        filename = f'data/custom_{min_num_nodes}_{max_num_nodes}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            Gs = [nx.ladder_graph(i) for i in range(10, 26)] + [nx.wheel_graph(i) for i in range(20, 51)] + [nx.cycle_graph(i) for i in range(20, 51)]+[nx.path_graph(i) for i in range(20, 51)]+[nx.star_graph(i) for i in range(19, 50)]

            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.n_max = 0
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample

            for G in Gs:
                if G.number_of_nodes() >= min_num_nodes and G.number_of_nodes() <= max_num_nodes:
                    adj = torch.from_numpy(nx.to_numpy_matrix(G)).float()
                    #L = nx.normalized_laplacian_matrix(G).toarray()
                    diags = np.sum(nx.to_numpy_matrix(G), axis=0)
                    diags = np.squeeze(np.asarray(diags))
                    D = sp.sparse.diags(diags).toarray()
                    L = D - nx.to_numpy_matrix(G)
                    with sp.errstate(divide="ignore"):
                        diags_sqrt = 1.0 / np.sqrt(diags)
                    diags_sqrt[np.isinf(diags_sqrt)] = 0
                    DH = sp.sparse.diags(diags).toarray()
                    L = np.linalg.multi_dot((DH, L, DH))
                    L = torch.from_numpy(L).float()
                    eigval, eigvec = torch.linalg.eigh(L)
                    
                    self.eigvals.append(eigval)
                    self.eigvecs.append(eigvec)
                    self.adjs.append(adj)
                    self.n_nodes.append(G.number_of_nodes())
                    if G.number_of_nodes() > self.n_max:
                        self.n_max = G.number_of_nodes()
                    max_eigval = torch.max(eigval)
                    if max_eigval > self.max_eigval:
                        self.max_eigval = max_eigval
                    min_eigval = torch.min(eigval)
                    if min_eigval < self.min_eigval:
                        self.min_eigval = min_eigval

            torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            last_idx = self.k if self.k < len(eigv) else len(eigv) - 1
            if eigv[last_idx] > self.max_k_eigval:
                self.max_k_eigval = eigv[last_idx].item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])

        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()

        return graph


def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm
    

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
