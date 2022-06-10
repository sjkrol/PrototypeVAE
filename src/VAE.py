'''
File containing VAE model class
'''

import torch
import torch.nn.functional as F
from torch import nn
from torchinfo import summary
from utils import list_of_distances

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        # initialise non model parameters
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.k1 = 0 # stores KL divergence

    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))

        z = mu + sigma*self.N.sample(mu.shape)
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
        # these help make sense of below, although interested in a different method for computing KL divergence
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) -1/2).sum()

        return z

class VariationalDecoder(nn.Module):
    def __init__(self, laten_dims):
        super(VariationalDecoder, self).__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(laten_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 3*3*32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))


        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
    
    def forward(self, x):

        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = VariationalDecoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        
        return self.decoder(z)

class Classifier(nn.Module):
    def __init__(self, latent_dims, n_prototypes):
        super(Classifier, self).__init__()

        self.prototypes = torch.randn((n_prototypes, latent_dims), requires_grad=True)
        self.logits = nn.Linear(n_prototypes, 10)
        self.prototype_distances = None
        self.feature_vector_distances = None
    
    def forward(self, x):

        # same order as paper
        self.prototype_distances = list_of_distances(x, self.prototypes)
        self.feature_vector_distances = list_of_distances(self.prototypes, x)

        x = self.logits(self.prototype_distances)
        
        return torch.sigmoid(x)

class PrototypeVAE(nn.Module):
    def __init__(self, latent_dims, n_prototypes):
        super(PrototypeVAE, self).__init__()

        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = VariationalDecoder(latent_dims)
        self.classifier = Classifier(latent_dims, n_prototypes)
    
    def forward(self, x):
        x = x.to(device)

        x = self.encoder(x)

        return self.decoder(x), self.classifier(x)


if __name__ == "__main__":

    model = PrototypeVAE(4, 5)
    batch_size = 16
    summary(model, input_size=(batch_size, 1, 28, 28))
    

