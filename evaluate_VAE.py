import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from DatasetConstruction.DataLoader import AlgaeBloomDataset
from Model.VAE import VAE
import numpy as np
from GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
Initialize Hyperparameters
"""

batch_size = 128
learning_rate = 1e-3
num_epochs = 100
device = 'cuda:1'



"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""
nav_map = np.genfromtxt('Maps/example_map.csv')
dataset = AlgaeBloomDataset(set='train')
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

dataset_test = AlgaeBloomDataset(set='test')
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

net = torch.load('Model/VAE_99.pt', map_location=device)

for i in range(10):

	with torch.no_grad():


		imgs, _ = dataset_test[i]
		gt_imgs = imgs[0, :, :].numpy()
		sensing_masks = imgs[1, :, :].numpy()
		# The input images are the first channel multiplied by the sensing mask
		imgs = gt_imgs * sensing_masks
		
		out, _ = net.imagine(imgs[np.newaxis, np.newaxis, :, :], N=0)
		
		out = out.squeeze(0).cpu().numpy()

		# Represent the input image and the output image in the same figure
		fig, axs = plt.subplots(1, 3, figsize=(12, 5))
		axs[0].imshow(imgs, cmap=algae_colormap, alpha=sensing_masks, vmin=0.0, vmax=1.0)
		axs[0].set_title('Input Image')

		axs[1].imshow(gt_imgs, cmap=algae_colormap, vmin=0.0, vmax=1.0, interpolation='bicubic')
		axs[1].set_title('Ground truth map')

		final_output = (out * (1-sensing_masks) + imgs * sensing_masks)*nav_map
		a = axs[2].imshow(final_output,  cmap=algae_colormap, vmin=0.0, vmax=1.0, interpolation='bicubic')
		divider = make_axes_locatable(axs[2])
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(a, cax=cax, orientation='vertical')

		axs[2].set_title('Predicted map')

		plt.tight_layout()

		plt.savefig('Results/VAE_Results/VAE_Results_{}.png'.format(i))
