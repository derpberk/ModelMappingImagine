from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from DatasetConstruction.DataLoader import AlgaeBloomDataset
from Model.VAE import VAE
import numpy as np
from GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap
from tensorboard import summary
from torch.utils.tensorboard import SummaryWriter


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
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

dataset_test = AlgaeBloomDataset(set='test')
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)


"""
Initialize the network and the Adam optimizer
"""
net = VAE(input_shape = (1, *dataset[0][0][0].shape),
            L_ce = 3.0,
            L_kl = 3.0,
            L_p = 1.0,
            L_r = 8.0).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


""" Create tensorboard writer """
writer = SummaryWriter(log_dir='runs/VAE', comment='VAE')

def publish_tensorboard_results(metrics, epoch, writer):

    for key in metrics.keys():
        for metric in metrics[key].keys():
            writer.add_scalar(f'{key}/{metric}', metrics[key][metric][-1], epoch)



"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""

metrics = {'train': {'loss': [], 'cross_entropy': [], 'kl_divergence': [], 'perceptual_loss': [], 'reconstruction_loss': [], 'mean_abs_error': [], 'mean_abs_error_masked': []},
              'test': {'loss': [], 'cross_entropy': [], 'kl_divergence': [], 'perceptual_loss': [], 'reconstruction_loss': [], 'mean_abs_error': [], 'mean_abs_error_masked': []},}
        

for epoch in range(num_epochs):

    # Initialize the loss
    train_loss = 0.0
    train_cross_entropy = 0.0
    train_kl_divergence = 0.0
    train_perceptual_loss = 0.0
    train_reconstruction_loss = 0.0
    train_mean_abs_error = 0.0
    train_mean_abs_error_masked = 0.0


    for idx, data in enumerate(train_loader, 0):

        imgs, _ = data

        with torch.no_grad():
            gt_imgs = imgs[:, 0, :, :].unsqueeze(1).to(device)
            sensing_masks = imgs[:, 1, :, :].unsqueeze(1).to(device)

            # The input images are the first channel multiplied by the sensing mask
            imgs = gt_imgs * sensing_masks

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(imgs)

        # Compute the loss
        loss, cross_entropy, kl_divergence, perceptual_loss, reconstruction_loss  = net.compute_loss(mu, logVar, imgs, out, gt_imgs, sensing_masks)

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

         # Append the loss to the metrics dictionary
        train_loss += loss.item()
        train_cross_entropy += cross_entropy.item()
        train_kl_divergence += kl_divergence.item()
        train_perceptual_loss += perceptual_loss.item()
        train_reconstruction_loss += reconstruction_loss.item()
        train_mean_abs_error += torch.mean(torch.abs(out - gt_imgs)).item()
        train_mean_abs_error_masked += torch.mean((1.0 - sensing_masks) * torch.abs(out - gt_imgs)).item()

    # Append the loss to the metrics dictionary
    metrics['train']['loss'].append(train_loss / len(train_loader))
    metrics['train']['cross_entropy'].append(train_cross_entropy / len(train_loader))
    metrics['train']['kl_divergence'].append(train_kl_divergence / len(train_loader))
    metrics['train']['perceptual_loss'].append(train_perceptual_loss / len(train_loader))
    metrics['train']['reconstruction_loss'].append(train_reconstruction_loss / len(train_loader))
    metrics['train']['mean_abs_error'].append(train_mean_abs_error / len(train_loader))
    metrics['train']['mean_abs_error_masked'].append(train_mean_abs_error_masked / len(train_loader))

    print('Epoch {}: Loss {} | cross_entropy: {} | kl_div : {} | perceptual_loss: {} | reconstruction loss: {}'.format(epoch, loss, cross_entropy, kl_divergence, perceptual_loss, reconstruction_loss))


    # Test the network on the test set
    with torch.no_grad():

        # Initialize the loss
        test_loss = 0.0
        test_cross_entropy = 0.0
        test_kl_divergence = 0.0
        test_perceptual_loss = 0.0
        test_reconstruction_loss = 0.0
        test_mean_abs_error = 0.0
        test_mean_abs_error_masked = 0.0

        for idx, data in enumerate(test_loader, 0):

            imgs, _ = data

            with torch.no_grad():
                gt_imgs = imgs[:, 0, :, :].unsqueeze(1).to(device)
                sensing_masks = imgs[:, 1, :, :].unsqueeze(1).to(device)

                # The input images are the first channel multiplied by the sensing mask
                imgs = gt_imgs * sensing_masks

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = net(imgs)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            loss, cross_entropy, kl_divergence, perceptual_loss, reconstruction_loss  = net.compute_loss(mu, logVar, imgs, out, gt_imgs, sensing_masks)


            # Append the loss to the metrics dictionary
            test_loss += loss.item()
            test_cross_entropy += cross_entropy.item()
            test_kl_divergence += kl_divergence.item()
            test_perceptual_loss += perceptual_loss.item()
            test_reconstruction_loss += reconstruction_loss.item()
            test_mean_abs_error += torch.mean(torch.abs(out - gt_imgs)).item()
            test_mean_abs_error_masked += torch.mean((1.0 - sensing_masks) * torch.abs(out - gt_imgs)).item()

        # Append the loss to the metrics dictionary
        metrics['test']['loss'].append(test_loss / len(test_loader))
        metrics['test']['cross_entropy'].append(test_cross_entropy / len(test_loader))
        metrics['test']['kl_divergence'].append(test_kl_divergence / len(test_loader))
        metrics['test']['perceptual_loss'].append(test_perceptual_loss / len(test_loader))
        metrics['test']['reconstruction_loss'].append(test_reconstruction_loss / len(test_loader))
        metrics['test']['mean_abs_error'].append(test_mean_abs_error / len(test_loader))
        metrics['test']['mean_abs_error_masked'].append(test_mean_abs_error_masked / len(test_loader))

    # Write metrics to tensorboard
    publish_tensorboard_results(metrics, epoch, writer)

    # Write images to tensorboard

    writer.add_image(f'Test/input', imgs[0, :, :, :], epoch)
    writer.add_image(f'Test/GT', gt_imgs[0, :, :, :], epoch)
    writer.add_image(f'Test/output', out[0, :, :, :], epoch)
    

torch.save(net, 'Model/VAE_{}.pt'.format(epoch))

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

