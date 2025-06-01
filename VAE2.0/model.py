import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    This model is a VAE for MNIST, which contains an encoder and a decoder.
    
    The encoder outputs mu_phi and log (sigma_phi)^2
    The decoder outputs mu_theta
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        Args:
            - input_dim: the input image, 1 * 28 * 28 = 784
            - hidden_dim: the dimension of the hidden layer of our MLP model
            - latent_dim: dimension of hidden vector z
        """
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  # mu_phi
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)  # log (sigma_phi)^2

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.act = nn.LeakyReLU(0.2)

    def encode(self, x):
        """ 
        Encode the image into z, representing q_phi(z|x) 
        
        Args:
            - x: the input image, we have to flatten it to (batchsize, 784) before input

        Output:
            - mu_phi, log (sigma_phi)^2
        """
        h = self.act(self.fc1(x))
        mu = self.fc2_mu(h)
        log_var = self.fc2_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """ Reparameterization trick """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        """ 
        Decode z into image x

        Args:
            - z: hidden code 
            - labels: the labels of the inputs, useless here
        
        Hint: During training, z should be reparameterized! While during inference, just sample a z from random.
        """
        h = self.act(self.fc3(z))
        recon_x =  torch.sigmoid(self.fc4(h))  # Using sigmoid to constrain the output to [0, 1]
        return recon_x.view(-1, 28, 28)

    def forward(self, x, labels):
        """ x: shape (batchsize, 28, 28) labels are not used here"""
        x = x.view(-1, 28 * 28)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, labels)
        return recon_x, mu, log_var
