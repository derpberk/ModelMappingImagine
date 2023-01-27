from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from DatasetConstruction.DataLoader import AlgaeBloomDataset
from Model.VAE import VAE
import numpy as np
from GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap
"""
Initialize Hyperparameters
"""

batch_size = 128
learning_rate = 1e-3
num_epochs = 100
device = 'cuda:0'


"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""
nav_map = np.genfromtxt('Maps/example_map.csv')
dataset = AlgaeBloomDataset(set='train')
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

dataset_test = AlgaeBloomDataset(set='test')
test_loader = DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=0)


"""
Initialize the network and the Adam optimizer
"""
net = VAE(input_shape = (1, *dataset[0][0][0].shape)).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""

for epoch in range(num_epochs):

    for idx, data in enumerate(train_loader, 0):

        imgs, _ = data

        with torch.no_grad():
            gt_imgs = imgs[:, 0, :, :].unsqueeze(1).to(device)
            sensing_masks = imgs[:, 1, :, :].unsqueeze(1).to(device)

            # The input images are the first channel multiplied by the sensing mask
            imgs = gt_imgs * sensing_masks

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(imgs)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.mean(-1 - logVar + mu.pow(2) + logVar.exp())
        perceptual_loss = torch.mean(torch.abs(out - gt_imgs))
        reconstruction_loss = torch.mean((1.0 - sensing_masks) * torch.abs(out - gt_imgs)) + torch.mean(sensing_masks * torch.abs(out - imgs))
        cross_entropy = F.binary_cross_entropy(out, imgs, size_average=True)

        loss = 1.0 * cross_entropy + 1.0 * kl_divergence + 5.0 * perceptual_loss + 5.0 * reconstruction_loss

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {}: Loss {} | cross_entropy: {} | kl_div : {} | perceptual_loss: {} | reconstruction loss: {}'.format(epoch, loss, cross_entropy, kl_divergence, perceptual_loss, reconstruction_loss))



VAE.save(net, 'Models/VAE_{}.pt'.format(epoch))

# Test the network on a single image and plot the output image next to the input image

import matplotlib.pyplot as plt



for i in range(10):

    with torch.no_grad():


        imgs, _ = dataset_test[i]
        gt_imgs = imgs[0].unsqueeze(0).unsqueeze(0).to(device)
        sensing_masks = imgs[1].unsqueeze(0).unsqueeze(0).to(device)

        # The input images are the first channel multiplied by the sensing mask
        imgs = gt_imgs * sensing_masks

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(imgs)

        # Represent the input image and the output image in the same figure
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(imgs[0, 0, :, :].cpu().numpy(), cmap=algae_colormap, alpha=sensing_masks[0, 0, :, :].cpu().numpy(), vmin=0.0, vmax=1.0)
        ax.set_title('Input Image')

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(gt_imgs[0, 0, :, :].cpu().numpy(), cmap=algae_colormap, vmin=0.0, vmax=1.0)
        ax.set_title('Ground Truth Image')

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(imgs[0, 0, :, :].cpu().numpy() * sensing_masks[0, 0, :, :].cpu().numpy() + (out[0, 0, :, :].cpu().numpy() * (1.0 - sensing_masks[0, 0, :, :].cpu().numpy())),  cmap=algae_colormap, vmin=0.0, vmax=1.0)
        ax.set_title('Output 1 Image')

        plt.savefig('Results/VAE_Results/VAE_Results_{}.png'.format(i))

