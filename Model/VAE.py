
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

"""
A Convolutional Variational Autoencoder
"""
class VAE(pl.LightningModule):
    def __init__(self, input_shape = (1, 64, 64), zDim=256, **kwargs):
        super(VAE, self).__init__()

        # Set the hyperparameters for weighting tyhe loss
        self.L_ce = kwargs['L_ce']
        self.L_kl = kwargs['L_kl']
        self.L_p  = kwargs['L_p']
        self.L_r  = kwargs['L_r']


        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, 8),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU()
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
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5),
            nn.ReLU(),
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

    def compute_loss(self, mu, logVar, input, out, gts, sensing_masks):

        # Compute the reconstruction loss and KL divergence loss
        # The loss is the summation of the two losses
        # The loss is used for backpropagation
        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt

        kl_divergence = 0.5 * torch.mean(-1 - logVar + mu.pow(2) + logVar.exp())
        perceptual_loss = torch.mean(torch.abs(out - gts))
        reconstruction_loss = torch.mean((1.0 - sensing_masks) * torch.abs(out - gts)) + torch.mean(sensing_masks * torch.abs(out - input))
        cross_entropy = F.binary_cross_entropy(out, input, reduction='mean')

        loss = self.L_ce * cross_entropy + self.L_ce * kl_divergence + self.L_ce * perceptual_loss + self.L_ce * reconstruction_loss

        return loss, cross_entropy, kl_divergence, perceptual_loss, reconstruction_loss

class lightning_VAE(pl.LightningModule):

    def __init__(self, input_shape = (1, 64, 64), zDim=256, **kwargs):
        super(lightning_VAE, self).__init__()

        self.model = VAE(input_shape = input_shape, zDim=zDim, **kwargs)

        self.L_ce = kwargs['L_ce']
        self.L_kl = kwargs['L_kl']
        self.L_p  = kwargs['L_p']
        self.L_r  = kwargs['L_r']

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, *args, **kwargs):
        """ Lightning calls this inside the training loop """

        # forward pass
        out, mu, logVar = self.model.forward(args[0])

        # compute loss
        loss, cross_entropy, kl_divergence, perceptual_loss, reconstruction_loss = self.model.compute_loss(mu, logVar, args[0], out, args[1], args[2])

        # log loss
        self.log('train_loss', loss)
        self.log('train_cross_entropy', cross_entropy)
        self.log('train_kl_divergence', kl_divergence)
        self.log('train_perceptual_loss', perceptual_loss)
        self.log('train_reconstruction_loss', reconstruction_loss)

        return loss

    def train(self):

        # train the model
        trainer = pl.Trainer(max_epochs=100, gpus=1)
        trainer.fit(self, self.train_dataloader(), self.val_dataloader())
        



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



    