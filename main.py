import argparse
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from autoencoder import AutoEncoder, VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import create_dataset, CustomDataset, linear_beta_schedule

np.random.seed(13)

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs-autoencoder', type=int, default=3000)
parser.add_argument('--hidden-dim-autoencoder', type=int, default=64)
parser.add_argument('--latent-dim', type=int, default=128)
parser.add_argument('--n-max-nodes', type=int, default=50)
parser.add_argument('--n-layers-autoencoder', type=int, default=2)
parser.add_argument('--spectral-emb-dim', type=int, default=10)
parser.add_argument('--variational-autoencoder', action='store_true', default=False)
parser.add_argument('--epochs-denoise', type=int, default=2000)
parser.add_argument('--timesteps', type=int, default=500)
parser.add_argument('--hidden-dim-denoise', type=int, default=256)
parser.add_argument('--n-layers_denoise', type=int, default=2)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Construct dataset
dataset = CustomDataset(k=args.spectral_emb_dim)

# Slit into training, validation and test sets
idx = np.random.permutation(len(dataset))
train_dataset = [dataset[idx[i]] for i in range(int(0.8*idx.size))]
val_dataset = [dataset[idx[i]] for i in range(int(0.8*idx.size), int(0.9*idx.size))]
test_dataset = [dataset[idx[i]] for i in range(int(0.9*idx.size), idx.size)]

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=False)

if args.variational_autoencoder:
    autoencoder = VariationalAutoEncoder(1, args.hidden_dim_autoencoder, args.latent_dim, args.n_layers_autoencoder, args.n_max_nodes).to(device)
else:
    autoencoder = AutoEncoder(1, args.hidden_dim_autoencoder, args.latent_dim, args.n_layers_autoencoder, args.n_max_nodes).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Train autoencoder
for epoch in range(1, args.epochs_autoencoder+1):
    autoencoder.train()
    train_loss_all = 0
    train_count = 0
    if args.variational_autoencoder:
        train_loss_all_recon = 0
        train_loss_all_kld = 0
    
    for data in train_loader:
        data['adj'] = data['adj'].to(device)
        data['mask'] = data['mask'].to(device)
        #data['eigvec'] = data['eigvec'][:,:,:args.spectral_emb_dim].to(device)
        degs = torch.bmm(data['adj'], torch.ones(data['adj'].size(0), data['adj'].size(1), 1, device=device))
        optimizer.zero_grad()
        if args.variational_autoencoder:
            loss, recon, kld  = autoencoder.loss_function(data['adj'], degs, data['mask'])
            train_loss_all_recon += data['adj'].size(0) * recon.item()
            train_loss_all_kld += data['adj'].size(0) * kld.item()
        else:
            loss = autoencoder.loss_function(data['adj'], degs, data['mask'])
        loss.backward()
        train_loss_all += data['adj'].size(0) * loss.item()
        train_count += data['adj'].size(0)
        optimizer.step()

    autoencoder.eval()
    val_loss_all = 0
    val_count = 0
    if args.variational_autoencoder:
        val_loss_all_recon = 0
        val_loss_all_kld = 0

    for data in val_loader:
        data['adj'] = data['adj'].to(device)
        data['mask'] = data['mask'].to(device)
        #data['eigvec'] = data['eigvec'][:,:,:args.spectral_emb_dim].to(device)
        degs = torch.bmm(data['adj'], torch.ones(data['adj'].size(0), data['adj'].size(1), 1, device=device))
        if args.variational_autoencoder:
            loss, recon, kld  = autoencoder.loss_function(data['adj'], degs, data['mask'])
            val_loss_all_recon += data['adj'].size(0) * recon.item()
            val_loss_all_kld += data['adj'].size(0) * kld.item()
        else:
            loss = autoencoder.loss_function(data['adj'], degs, data['mask'])
        val_loss_all += data['adj'].size(0) * loss.item()
        val_count += data['adj'].size(0)

    if epoch % 5 == 0:
        if args.variational_autoencoder:
            print('Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(epoch, train_loss_all/train_count, train_loss_all_recon/train_count, train_loss_all_kld/train_count, val_loss_all/val_count, val_loss_all_recon/val_count, val_loss_all_kld/val_count))
        else:
            print('Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, train_loss_all/train_count, val_loss_all/val_count))

    scheduler.step()

torch.save({
    'state_dict': autoencoder.state_dict(),
    'optimizer' : optimizer.state_dict(),
}, 'autoencoder.pth.tar')

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

denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Train denoising model
for epoch in range(1, args.epochs_denoise+1):
    denoise_model.train()
    train_loss_all = 0
    train_count = 0
    for data in train_loader:
        data['adj'] = data['adj'].to(device)
        data['mask'] = data['mask'].to(device)
        #data['eigvec'] = data['eigvec'][:,:,:args.spectral_emb_dim].to(device)
        degs = torch.bmm(data['adj'], torch.ones(data['adj'].size(0), data['adj'].size(1), 1, device=device))
        optimizer.zero_grad()
        x_g = autoencoder.encode(data['adj'], degs, data['mask'])
        t = torch.randint(0, args.timesteps, (data['adj'].size(0),), device=device).long()
        loss = p_losses(denoise_model, x_g, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
        loss.backward()
        train_loss_all += data['adj'].size(0) * loss.item()
        train_count += data['adj'].size(0)
        optimizer.step()

    denoise_model.eval()
    val_loss_all = 0
    val_count = 0
    for data in val_loader:
        data['adj'] = data['adj'].to(device)
        data['mask'] = data['mask'].to(device)
        #data['eigvec'] = data['eigvec'][:,:,:args.spectral_emb_dim].to(device)
        degs = torch.bmm(data['adj'], torch.ones(data['adj'].size(0), data['adj'].size(1), 1, device=device))
        x_g = autoencoder.encode(data['adj'], degs, data['mask'])
        t = torch.randint(0, args.timesteps, (data['adj'].size(0),), device=device).long()
        loss = p_losses(denoise_model, x_g, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
        val_loss_all += data['adj'].size(0) * loss.item()
        val_count += data['adj'].size(0)

    if epoch % 5 == 0:
        print('Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, train_loss_all/train_count, val_loss_all/val_count))

    scheduler.step()

torch.save({
    'state_dict': denoise_model.state_dict(),
    'optimizer' : optimizer.state_dict(),
}, 'denoise_model.pth.tar')

# sample 64 graphs
samples = sample(denoise_model, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=args.batch_size)

# show a random one
random_index = 5
x_g_denoise = autoencoder.decode(samples[-1])
print(torch.sum(x_g_denoise[random_index,:,:], dim=1))
