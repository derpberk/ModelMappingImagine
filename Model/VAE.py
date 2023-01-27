
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, input_shape = (1, 64, 64), zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, 8),
            nn.GELU(),
            nn.Conv2d(16, 32, 5),
            nn.GELU(),
            nn.Conv2d(32, 64, 3),
            nn.GELU()
        )


        # Get the size of the output of the last convolutional layer
        self.out_size = self.encoder(torch.zeros(1, *input_shape)).shape

        hidden_dim_size = torch.prod(torch.tensor(self.encoder(torch.zeros(1, *input_shape)).shape))

        self.encFC1 = nn.Linear(hidden_dim_size, zDim)
        self.encFC2 = nn.Linear(hidden_dim_size, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, hidden_dim_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 5),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, 8),
            nn.Sigmoid()
        )

    def encode(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = self.encoder(x)
        x = x.flatten(1)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.reshape(-1, self.out_size[1], self.out_size[2], self.out_size[3])
        x = self.decoder(x)

        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        out = self.decode(z)
        return out, mu, logVar

if __name__ == '__main__':
    # Test the VAE

    # load matrix of the map 
    import numpy as np

    map = np.genfromtxt('../Maps/example_map.csv')

    input_shape = (1, *map.shape)
    model = VAE(input_shape = input_shape, zDim=256)
    print(model)

    # Test the forward pass
    x = torch.randn(1, *input_shape)
    out, mu, logVar = model.forward(x)
    print(out.shape, mu.shape, logVar.shape)

    # Test the reparameterization
    z = model.reparameterize(mu, logVar)
    print(z.shape)

    # Test the decoder
    out = model.decode(z)
    print(out.shape)

    # Forward a random image and represent the input and the output side by side
    import matplotlib.pyplot as plt

    x = torch.FloatTensor(map).unsqueeze(0).unsqueeze(0)
    out, mu, logVar = model.forward(x)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.gca().set_title('Input')
    plt.imshow(x[0, 0, :, :].detach().numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.gca().set_title('Output')
    plt.imshow(out[0, 0, :, :].detach().numpy(), cmap='gray')
    plt.show()



    