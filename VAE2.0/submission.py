import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: 2.2 Your VAE model here!
class VAE(nn.Module):
    """
    This model is a VAE for MNIST, which contains an encoder and a decoder.
    
    The encoder outputs mu_phi and log (sigma_phi)^2
    The decoder outputs mu_theta
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        You should define your model parameters and the network architecture here.
        """
        super(VAE, self).__init__()
        
        # TODO: 2.2.1 Define your encoder and decoder
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
        # TODO: 2.2.2 finish the encode code, input is x, output is mu_phi and log(sigma_theta)^2
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
        # TODO: 2.2.3 finish the decoding code, input is z, output is recon_x or mu_theta
        # Hint: output should be within [0, 1], maybe you can use torch.sigmoid()
        h = self.act(self.fc3(z))
        recon_x = torch.sigmoid(self.fc4(h))  # Using sigmoid to constrain the output to [0, 1]
        return recon_x.view(-1, 28, 28)

    def forward(self, x, labels):
        """ x: shape (batchsize, 28, 28) labels are not used here"""
        # TODO: 2.2.4 passing the whole model, first encoder, then decoder, output all we need to cal loss
        # Hint1: all input data is [0, 1], 
        # and input tensor's shape is [batch_size, 1, 28, 28], 
        # maybe you have to change the shape to [batch_size, 28 * 28] if you use MLP model using view()
        # Hint2: maybe 3 or 4 lines of code is OK!
        # x = x.view(-1, 28 * 28)
        x = x.view(-1, 28 * 28)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, labels)
        return recon_x, mu, log_var

# TODO: 2.3 Calculate vae loss using input and output
def vae_loss(recon_x, x, mu, log_var, var=0.5):
    batch_size = x.size(0)
    recon_loss = F.mse_loss(recon_x.view(batch_size, -1), x.view(batch_size, -1), reduction='none').sum(dim=1).mean()
    kl_loss = -0.5 * torch.sum((1 + log_var - mu.pow(2) - log_var.exp()), dim=1).mean()
    loss = recon_loss + kl_loss
    return loss

# TODO: 3 Design the model to finish generation task using label
class GenModel(nn.Module):
    pass