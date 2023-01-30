""" Optuna for optimizing the hyperparameters of the VAE model. """

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from DatasetConstruction.DataLoader import AlgaeBloomDataset
from Model.VAE import VAE
import numpy as np
from tqdm import trange



""" Objective function for optimizing the hyperparameters of the VAE model. """

def objective(trial: optuna.trial.Trial) -> float:
	
	device = 'cuda:0'

	""" Sample hyperparameters from the search space. """
	# Suggest values for the batch size
	# Suggest floats for L_ce, L_kl, L_p, L_r
	L_ce = trial.suggest_float("L_ce", 0.1, 10.0)
	L_kl = trial.suggest_float("L_kl", 0.1, 10.0)
	L_p = trial.suggest_float("L_p", 0.1, 10.0)
	L_r = trial.suggest_float("L_r", 0.1, 10.0)


	""" Load the dataset """
	nav_map = np.genfromtxt('Maps/example_map.csv')
	dataset = AlgaeBloomDataset(set='train')
	train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

	dataset_test = AlgaeBloomDataset(set='test')
	test_loader = DataLoader(dataset_test, batch_size=128, shuffle=True, num_workers=0)

	# Construct dict of hyperparameters #
	L_vae = {'L_ce': L_ce, 'L_kl': L_kl, 'L_p': L_p, 'L_r': L_r}

	""" Instantiate the model """
	net = VAE(input_shape = (1, *dataset[0][0][0].shape, L_vae),
			L_ce = 3.0,
			L_kl = 3.0,
			L_p = 1.0,
			L_r = 8.0).to('cuda:0')

	optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

	""" Loop over the epochs """
	for epoch in trange(1, 100 + 1):

		net.train()
		train_loss = 0
		
		for batch_idx, data in enumerate(train_loader):

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

			train_loss += loss.item()

			
	# Evaluate the model on the test set
	net.eval()

	error = 0

	for _, data in enumerate(test_loader, 0):

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
			
			# Accumulate the error
			error += torch.mse_loss(out, gt_imgs).item()

	# Return the error

	return error


if __name__ == "__main__":

	pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

	study = optuna.create_study(direction="maximize", pruner=pruner)
	study.optimize(objective, n_trials=50, timeout=600, show_progress_bar=True)

	print("Number of finished trials: {}".format(len(study.trials)))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: {}".format(trial.value))

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))