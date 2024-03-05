import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F
import community as community_louvain

from torch import Tensor
from torch.utils.data import Dataset

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram

def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G


def eval_autoencoder(test_loader, autoencoder, n_max_nodes, device):
    Gs = []
    Gs_rec = []
    sims = []
    for data in test_loader:
        Gs = []
        Gs_rec = []

        data = data.to(device)
        adj_rec = autoencoder(data)
        adj_rec[adj_rec>0.5] = 1
        adj_rec[adj_rec<=0.5] = 0

        for i in range(data.A.size(0)):
            Gs.append(construct_nx_from_adj(data.A[i,:,:].detach().cpu().numpy()))
            Gs_rec.append(construct_nx_from_adj(adj_rec[i,:,:].detach().cpu().numpy()))

        for G in Gs:
            for node in G.nodes():
                G.nodes[node]['label'] = 1

        for G in Gs_rec:
            for node in G.nodes():
                G.nodes[node]['label'] = 1

        Gs_pairs = [graph_from_networkx([Gs[i], Gs_rec[i]], node_labels_tag='label') for i in range(len(Gs))]
        wl_kernel = WeisfeilerLehman(n_iter=3, normalize=True, base_graph_kernel=VertexHistogram)

        for i in range(len(Gs_pairs)):
            K = wl_kernel.fit_transform(Gs_pairs[i])
            sims.append(K[0,1])

    print('Average similarity:', np.mean(sims))


def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x


def read_stats(file):
    stats = []
    fread = open(file, "r")
    #print(file)
    for i,line in enumerate(fread):
        if i == 13: continue
        line = line.strip()
        tokens = line.split(":")
        #print(tokens[-1])
        #stats.append(handle_nan(float(tokens[-1].strip())))
        stats.append(float(tokens[-1].strip()))
    fread.close()
    return stats



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




def calculate_stats_graph(G):
    stats = []
    # Number of nodes
    num_nodes = handle_nan(float(G.number_of_nodes()))
    stats.append(num_nodes)
    # Number of edges
    num_edges = handle_nan(float(G.number_of_edges()))
    stats.append(num_edges)
    # Density
    density = handle_nan(float(nx.density(G)))
    stats.append(density)
    # Degree statistics
    degrees = [deg for node, deg in G.degree()]
    max_degree = handle_nan(float(max(degrees)))
    stats.append(max_degree)
    min_degree = handle_nan(float(min(degrees)))
    stats.append(min_degree)
    avg_degree = handle_nan(float(sum(degrees) / len(degrees)))
    stats.append(avg_degree)
    # Assortativity coefficient
    assortativity = handle_nan(float(nx.degree_assortativity_coefficient(G)))
    stats.append(assortativity)
    # Number of triangles
    triangles = nx.triangles(G)
    num_triangles = handle_nan(float(sum(triangles.values()) // 3))
    stats.append(num_triangles)
    # Average number of triangles formed by an edge
    avg_triangles = handle_nan(float(sum(triangles.values()) / num_edges))
    stats.append(avg_triangles)
    # Maximum number of triangles formed by an edge
    max_triangles_per_edge = handle_nan(float(max(triangles.values())))
    stats.append(max_triangles_per_edge)
    # Average local clustering coefficient
    avg_clustering_coefficient = handle_nan(float(nx.average_clustering(G)))
    stats.append(avg_clustering_coefficient)
    # Global clustering coefficient
    global_clustering_coefficient = handle_nan(float(nx.transitivity(G)))
    stats.append(global_clustering_coefficient)
    # Maximum k-core
    max_k_core = handle_nan(float(max(nx.core_number(G).values())))
    stats.append(max_k_core)
    # Lower bound of Maximum Clique
    #lower_bound_max_clique = handle_nan(float(nx.graph_clique_number(G)))
    #stats.append(lower_bound_max_clique)

    # calculate communities
    partition = community_louvain.best_partition(G)
    n_communities = handle_nan(float(len(set(partition.values()))))
    stats.append(n_communities)

    # calculate diameter
    connected_components = list(nx.connected_components(G))
    # Initialize diameter to a small value
    diameter = float(0)
    # Iterate over connected components and find the maximum diameter
    for component in connected_components:
        subgraph = G.subgraph(component)
        component_diameter = nx.diameter(subgraph)
        diameter = handle_nan(float(max(diameter, component_diameter)))
    stats.append(diameter)
    return stats

def store_stats(y, y_pred, fw_name1, fw_name2):
    fw1 = open(fw_name1,"w")
    fw2 = open(fw_name2,"w")

    for el in y:
        np.savetxt(fw1, el, newline=' ')
        fw1.write('\n')
    fw1.close()

    for el in y_pred:
        np.savetxt(fw2, el, newline=' ')
        fw2.write('\n')
    fw2.close()




def gen_stats(G):
    y_pred = calculate_stats_graph(G)
    y_pred = np.nan_to_num(y_pred, nan=-100.0)
    return y_pred


def precompute_missing(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    y = np.nan_to_num(y, nan=-100.0)
    y_pred = np.nan_to_num(y_pred, nan=-100.0)
    # Find indices where y is -100
    indices_to_change = np.where(y == -100.0)

    # Set corresponding elements in y and y_pred to 0
    y[indices_to_change] = 0.0
    y_pred[indices_to_change] = 0.0
    zeros_per_column = np.count_nonzero(y, axis=0)

    list_from_array = zeros_per_column.tolist()
    dc = {}
    for i in range(len(list_from_array)):
        dc[i] = list_from_array[i]
    return dc, y, y_pred



def sum_elements_per_column(matrix, dc):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    column_sums = [0] * num_cols

    for col in range(num_cols):
        for row in range(num_rows):
            column_sums[col] += matrix[row][col]

    res = []
    for col in range(num_cols):
        x = column_sums[col]/dc[col]
        res.append(x)

    return res



def calculate_mean_std(x):

    sm = [0 for i in range(15)]
    samples = [0 for i in range(15)]

    for el in x:
        for i, it in enumerate(el):
            if not math.isnan(it):
                sm[i] += it
                samples[i] += 1

    mean = [k / y for k,y in zip(sm, samples)]


    sm2 = [0 for i in range(16)]

    std = []

    for el in x:
        for i, it in enumerate(el):
            if not math.isnan(it):
                k = (it - mean[i])**2
                sm2[i] += k

    std = [(k / y)**0.5 for k,y in zip(sm2, samples)]
    return mean, std



def evaluation_metrics(y, y_pred, eps=1e-10):
    dc, y, y_pred = precompute_missing(y, y_pred)

    mse_st = (y - y_pred) ** 2
    mae_st = np.absolute(y - y_pred)

    mse = sum_elements_per_column(mse_st, dc)
    mae = sum_elements_per_column(mae_st, dc)

    #mse = [sum(x)/len(mse_st) for x in zip(*mse_st)]
    #mae = [sum(x)/len(mae_st) for x in zip(*mae_st)]

    a = np.absolute(y - y_pred)
    b = np.absolute(y) + np.absolute(y_pred)+ eps
    norm_error_st = (a/b)

    norm_error = sum_elements_per_column(norm_error_st, dc)
    #[sum(x)*100/len(norm_error_st) for x in zip(*norm_error_st)]

    return mse, mae, norm_error


def z_score_norm(y, y_pred, mean, std, eps=1e-10):

    y = np.array(y)
    y_pred = np.array(y_pred)

    normalized_true = (y - mean) / std

    normalized_gen = (y_pred - mean) / std

    dc, normalized_true, normalized_gen = precompute_missing(normalized_true, normalized_gen)

    #print(np.isnan(normalized_true).any())
    #print(np.isnan(normalized_gen).any())

    # Calculate MSE using normalized tensors
    mse_st = (normalized_true - normalized_gen) ** 2
    mae_st = np.absolute(normalized_true - normalized_gen)

    mse = sum_elements_per_column(mse_st, dc)
    mae = sum_elements_per_column(mae_st, dc)

    mse = np.sum(mse)/15
    mae = np.sum(mae)/15

    a = np.absolute(normalized_true - normalized_gen)
    b = np.absolute(normalized_true) + np.absolute(normalized_gen) + eps
    norm_error_st = (a/b)
    norm_error = sum_elements_per_column(norm_error_st, dc)
    norm_error = np.sum(norm_error)/15


    return mse, mae, norm_error
