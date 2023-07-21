import torch
import torch.nn as nn
import torch.nn.functional as F

from ppgn import Powerful

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
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
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, n_max_nodes):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = Powerful(num_layers=2, input_features=input_dim+1, hidden=hidden_dim, hidden_final=hidden_dim, dropout_prob=0.2, simplified=False, n_nodes=n_max_nodes, output_features=latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, n_layers, n_max_nodes)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, A, node_features, mask):
        x_g = self.encoder(A, node_features, mask)
        adj = self.decoder(x_g)
        return adj

    def encode(self, A, node_features, mask):
        x_g = self.encoder(A, node_features, mask)
        return x_g

    def decode(self, x_g):
       adj = self.decoder(x_g)
       return adj

    def loss_function(self, A, node_features, mask, beta=0.05):
        x_g  = self.encoder(A, node_features, mask)
        adj = self.decoder(x_g)
        return F.l1_loss(adj, A)

# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = Powerful(num_layers=2, input_features=input_dim+1, hidden=hidden_dim, hidden_final=hidden_dim, dropout_prob=0.2, simplified=False, n_nodes=n_max_nodes, output_features=latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, n_layers, n_max_nodes)
        self.fc = nn.Linear(hidden_dim, 1)
        #self.softsort = SoftSort()

    def forward(self, A, node_features, mask):
        x_g = self.encoder(A, node_features, mask)
        adj = self.decoder(x_g)
        return adj

    def encode(self, A, node_features, mask):
        x_g = self.encoder(A, node_features, mask)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def loss_function(self, A, node_features, mask, beta=0.05):
        x_g  = self.encoder(A, node_features, mask)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        
        # degs = torch.bmm(adj, torch.ones(adj.size(0), adj.size(1), 1, device=adj.device))
        # _, x_n_gen = self.encoder(adj, degs, torch.ones(adj.size(), device=adj.device))
        # scores = self.fc(x_n_gen).squeeze()
        # scores = torch.reshape(scores, (adj.size(0), adj.size(1)))
        # P = self.softsort(scores)
        # adj = torch.einsum("abc,acd->abd", (P, adj))
        # adj = torch.einsum("abc,adc->abd", (adj, P))
            
        # scores = self.fc(x_n).squeeze()
        # scores = torch.reshape(scores, (A.size(0), A.size(1)))
        # P = self.softsort(scores)
        # A = torch.einsum("abc,acd->abd", (P, A))
        # A = torch.einsum("abc,adc->abd", (A, P))

        recon = F.l1_loss(adj, A, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld