from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from DatasetConstruction.DataLoader import AlgaeBloomDataset
from Model.VAE import VAE
import numpy as np
from GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap
from torch.utils.tensorboard import SummaryWriter


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


"""
Initialize the network and the Adam optimizer
"""
net = VAE(input_shape = (1, *dataset[0][0][0].shape),
		  L_kl = 0.20797896567563687,
		  L_p = 7.854821467338043,
		  L_r = 5.707341393506469,
		  L_ce = 5.0,
		  L_dfc = 8.985864458775279,
		  device=device,
		  deep_feature_coherent=True).to(device)

net.set_optimizer()


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

metrics = {'train': {'loss': [], 'features_loss': [], 'kl_divergence': [], 'perceptual_loss': [], 'reconstruction_loss': []},
			  'test': {'loss': [], 'features_loss': [], 'kl_divergence': [], 'perceptual_loss': [], 'reconstruction_loss': []},}
		

for epoch in range(num_epochs):

	# Initialize the loss
	train_loss = 0.0
	train_feature_loss = 0.0
	train_kl_divergence = 0.0
	train_perceptual_loss = 0.0
	train_reconstruction_loss = 0.0
	net.train()

	for idx, data in enumerate(train_loader, 0):

		imgs, _ = data

		loss, kl_divergence, perceptual_loss, reconstruction_loss, features_loss = net.train_batch(imgs)

		 # Append the loss to the metrics dictionary
		train_loss += loss
		train_feature_loss += features_loss
		train_kl_divergence += kl_divergence
		train_perceptual_loss += perceptual_loss
		train_reconstruction_loss += reconstruction_loss

	# Append the loss to the metrics dictionary
	metrics['train']['loss'].append(train_loss / len(train_loader))
	metrics['train']['features_loss'].append(train_feature_loss / len(train_loader))
	metrics['train']['kl_divergence'].append(train_kl_divergence / len(train_loader))
	metrics['train']['perceptual_loss'].append(train_perceptual_loss / len(train_loader))
	metrics['train']['reconstruction_loss'].append(train_reconstruction_loss / len(train_loader))

	print('Epoch {}: Loss {} | features_loss: {} | kl_div : {} | perceptual_loss: {} | reconstruction loss: {}'.format(epoch, loss, features_loss, kl_divergence, perceptual_loss, reconstruction_loss))


torch.save(net, 'Model/VAE_{}.pt'.format(epoch))


