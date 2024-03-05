import argparse
import os
import random
import scipy as sp

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import AutoEncoder, VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import create_dataset, CustomDataset, linear_beta_schedule, read_stats, eval_autoencoder, construct_nx_from_adj, store_stats, gen_stats, calculate_mean_std, evaluation_metrics, z_score_norm

from torch.utils.data import Subset
np.random.seed(13)

# TODO: check/count number of all parameters

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs-autoencoder', type=int, default=200)
parser.add_argument('--hidden-dim-encoder', type=int, default=64)
parser.add_argument('--hidden-dim-decoder', type=int, default=256)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--n-max-nodes', type=int, default=100)
parser.add_argument('--n-layers-encoder', type=int, default=2)
parser.add_argument('--n-layers-decoder', type=int, default=3)
parser.add_argument('--spectral-emb-dim', type=int, default=10)
parser.add_argument('--variational-autoencoder', action='store_true', default=True)
parser.add_argument('--epochs-denoise', type=int, default=100)
parser.add_argument('--timesteps', type=int, default=500)
parser.add_argument('--hidden-dim-denoise', type=int, default=512)
parser.add_argument('--n-layers_denoise', type=int, default=3)
parser.add_argument('--train-autoencoder', action='store_false', default=True)
parser.add_argument('--train-denoiser', action='store_true', default=True)
parser.add_argument('--n-properties', type=int, default=15)
parser.add_argument('--dim-condition', type=int, default=128)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

graph_types  = ["barabasi_albert", "cycle", "dual_barabasi_albert", "extended_barabasi_albert", "fast_gnp","ladder", "lobster", "lollipop","newman_watts_strogatz","regular", "partition","path", "powerlaw","star","stochastic","watts_strogatz","wheel"]

gr2id = {graph_types[i]:i for i in range(len(graph_types))}


data_lst = []
filename = f'data/generated_dataset.pt'

stats_lst = []
# traverse through all the graphs of the folder
files = [f for f in os.listdir("generated data/graphs")]
#files_stats = [f for f in os.listdir("/data/iakovos/Multimodal/generated data/stats")]
print(len(files))
if os.path.isfile(filename):
    data_lst = torch.load(filename)
    print(f'Dataset {filename} loaded from file')

else:
    adjs = []
    eigvals = []
    eigvecs = []
    n_nodes = []
    max_eigval = 0
    min_eigval = 0
    for fileread in tqdm(files):
        tokens = fileread.split("/")
        idx = tokens[-1].find(".")
        filen = tokens[-1][:idx]
        extension = tokens[-1][idx+1:]
        # filename = f'data/'+filen+'.pt'
        #self.ignore_first_eigv = ignore_first_eigv
        fread = os.path.join("generated data/graphs",fileread)
        fstats = os.path.join("generated data/stats",filen+".txt")
        type = None
        for t in graph_types:
            if t in filen:
                type = t
        type_id = gr2id[type]
        #load dataset to networkx
        if extension == "gml":
            G = nx.read_gml(fread)
        else:
            G = nx.read_gexf(fread)
        # use canonical order (BFS) to create adjacency matrix
        ### BFS & DFS from largest-degree node
        stats_lst.append(fstats)
        CGs = [G.subgraph(c) for c in nx.connected_components(G)]

        # rank connected componets from large to small size
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

        node_list_bfs = []
        #node_list_dfs = []
        for ii in range(len(CGs)):
          node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
          degree_sequence = sorted(
              node_degree_list, key=lambda tt: tt[1], reverse=True)

          bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
          #dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

          node_list_bfs += list(bfs_tree.nodes())
          #node_list_dfs += list(dfs_tree.nodes())

        adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

        adj = torch.from_numpy(adj_bfs).float()
        #L = nx.normalized_laplacian_matrix(G).toarray()
        diags = np.sum(adj_bfs, axis=0)
        diags = np.squeeze(np.asarray(diags))
        D = sparse.diags(diags).toarray()
        L = D - adj_bfs
        with sp.errstate(divide="ignore"):
            diags_sqrt = 1.0 / np.sqrt(diags)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        DH = sparse.diags(diags).toarray()
        L = np.linalg.multi_dot((DH, L, DH))
        L = torch.from_numpy(L).float()
        eigval, eigvecs = torch.linalg.eigh(L)
        eigval = torch.real(eigval)
        eigvecs = torch.real(eigvecs)
        idx = torch.argsort(eigval)
        eigvecs = eigvecs[:,idx]

        edge_index = torch.nonzero(adj).t()

        size_diff = args.n_max_nodes - G.number_of_nodes()
        x = torch.zeros(G.number_of_nodes(), args.spectral_emb_dim+1)
        x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(args.n_max_nodes-1)
        mn = min(G.number_of_nodes(),args.spectral_emb_dim)
        mn+=1
        x[:,1:mn] = eigvecs[:,:args.spectral_emb_dim]
        #print(x.size())
        adj = F.pad(adj, [0, size_diff, 0, size_diff])
        adj = adj.unsqueeze(0)
        #A = torch.zeros(1, args.n_max_nodes, args.n_max_nodes, args.spectral_emb_dim+2)
        #A[0,:,:,0] = adj
        #for i in range(G.number_of_nodes()):
        #    A[0,i,i,1:] = x[i,:]
        feats_stats = read_stats(fstats)
        feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
        data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, class=type, class_label=type_id))
    torch.save(data_lst, filename)
    print(f'Dataset {filename} saved')


# Slit into training, validation and test sets
idx = np.random.permutation(len(data_lst))
train_size = int(0.8*idx.size)
val_size = int(0.1*idx.size)

train_idx = [int(i) for i in idx[:train_size]]
val_idx = [int(i) for i in idx[train_size:train_size + val_size]]
test_idx = [int(i) for i in idx[train_size + val_size:]]

train_loader = DataLoader([data_lst[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader([data_lst[i] for i in val_idx], batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader([data_lst[i] for i in test_idx], batch_size=args.batch_size, shuffle=False)

if args.variational_autoencoder:
    autoencoder = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)
else:
    autoencoder = AutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


trainable_params_autoenc = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
print("Number of Autoencoder's trainable parameters: "+str(trainable_params_autoenc))

# Train autoencoder
if args.train_autoencoder:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder+1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        if args.variational_autoencoder:
            train_loss_all_recon = 0
            train_loss_all_kld = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            if args.variational_autoencoder:
                loss, recon, kld  = autoencoder.loss_function(data)
                train_loss_all_recon += recon.item()
                train_loss_all_kld += kld.item()
            else:
                loss = autoencoder.loss_function(data)#*data.x.size(0)
            loss.backward()
            if args.variational_autoencoder:
                train_loss_all += loss.item()
            else:
                train_loss_all += (torch.max(data.batch)+1) * loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        if args.variational_autoencoder:
            val_loss_all_recon = 0
            val_loss_all_kld = 0

        for data in val_loader:
            data = data.to(device)
            if args.variational_autoencoder:
                loss, recon, kld  = autoencoder.loss_function(data)
                val_loss_all_recon += recon.item()
                val_loss_all_kld += kld.item()
            else:
                loss = autoencoder.loss_function(data)#*data.x.size(0)
            if args.variational_autoencoder:
                val_loss_all += loss.item()
            else:
                val_loss_all += torch.max(data.batch)+1 * loss.item()
            val_count += torch.max(data.batch)+1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if args.variational_autoencoder:
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(dt_t,epoch, train_loss_all/train_count, train_loss_all_recon/train_count, train_loss_all_kld/train_count, val_loss_all/val_count, val_loss_all_recon/val_count, val_loss_all_kld/val_count))
            else:
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'autoencoder.pth.tar')
else:
    checkpoint = torch.load('autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()
eval_autoencoder(test_loader, autoencoder, args.n_max_nodes, device) # add also mse (loss that we use generally)


# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

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

denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_properties, d_cond=args.dim_condition).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

trainable_params_diff = sum(p.numel() for p in denoise_model.parameters() if p.requires_grad)
print("Number of Diffusion model's trainable parameters: "+str(trainable_params_diff))

if args.train_denoiser:
    # Train denoising model
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'denoise_model.pth.tar')
else:
    checkpoint = torch.load('denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])

denoise_model.eval()

del train_loader, val_loader


ground_truth = []
pred = []


for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
    data = data.to(device)
    stat = data.stats
    bs = stat.size(0)
    samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
    x_sample = samples[-1]
    adj = autoencoder.decode_mu(x_sample)
    stat_d = torch.reshape(stat, (-1, args.n_properties))

    for i in range(stat.size(0)):
        #adj = autoencoder.decode_mu(samples[random_index])
        # Gs_generated.append(construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy()))
        stat_x = stat_d[i]

        Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
        stat_x = stat_x.detach().cpu().numpy()
        ground_truth.append(stat_x)
        pred.append(gen_stats(Gs_generated))


store_stats(ground_truth, pred, "y_stats.txt", "y_pred_stats.txt")

# stats = torch.cat(stats, dim=0).detach().cpu().numpy()


mean, std = calculate_mean_std(ground_truth)


mse, mae, norm_error = evaluation_metrics(ground_truth, y_pred)


mse_all, mae_all, norm_error_all, mean_perc_error_all = z_score_norm(ground_truth, pred, mean, std)



feats_lst = ["number of nodes", "number of edges", "density","max degree", "min degree", "avg degree","assortativity","triangles","avg triangles","max triangles","avg clustering coef", "global clustering coeff", "max k-core", "communities","diameter"]
id2feats = {i:feats_lst[i] for i in range(len(mse))}




print("MSE for the samples in all features is equal to: "+str(mse_all))
print("MAE for the samples in all features is equal to: "+str(mae_all))
print("Symmetric Mean absolute Percentage Error for the samples for all features is equal to: "+str(norm_error_all*100))
print("=" * 100)

for i in range(len(mse)):
    print("MSE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mse[i]))
    print("MAE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mae[i]))
    print("Symmetric Mean absolute Percentage Error for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(norm_error[i]*100))
    print("=" * 100)
