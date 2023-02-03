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

EPOCHS = 100
TRIALS = 25

""" Objective function for optimizing the hyperparameters of the VAE model. """

def objective(trial: optuna.trial.Trial) -> float:
	
	device = 'cuda:1'

	""" Sample hyperparameters from the search space. """
	# Suggest values for the batch size
	# Suggest floats for L_ce, L_kl, L_p, L_r
	L_kl = trial.suggest_float("L_kl", 0.1, 10.0)
	L_p = trial.suggest_float("L_p", 0.1, 10.0)
	L_r = trial.suggest_float("L_r", 0.1, 10.0)
	L_dfc = trial.suggest_float("L_dfc", 0.1, 10.0)
	L_ce = trial.suggest_float("L_ce", 0.1, 10.0)


	""" Load the dataset """
	nav_map = np.genfromtxt('Maps/example_map.csv')
	dataset = AlgaeBloomDataset(set='train')
	train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

	dataset_test = AlgaeBloomDataset(set='test')
	test_loader = DataLoader(dataset_test, batch_size=128, shuffle=True, num_workers=0)

	""" Instantiate the model """
	net = VAE(input_shape = (1, *dataset[0][0][0].shape), deep_feature_coherent=True, L_dfc = L_dfc, L_kl=L_kl, L_p=L_p, L_r=L_r, L_ce = L_ce, device = device).to(device)

	""" Set the optimizer """
	net.set_optimizer()

	""" Loop over the epochs """

	for epoch in trange(0, EPOCHS):
		
		# TIME TO TRAIN
		net.train()
		train_loss = 0
		
		for _, data in enumerate(train_loader):
			imgs, _ = data
			loss, _, _, _, _ = net.train_batch(imgs)
			train_loss += loss
			
		# TIME TO TEST
		net.eval()
		error = 0
		for _, data in enumerate(test_loader, 0):
				imgs, _ = data
				mse = net.evaluate_batch(imgs, N = 0)
				error += mse

		# Report the error
		trial.report(error, epoch)
		# Check if the trial should be pruned
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	# Return the error
	return error


if __name__ == "__main__":

	pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

	study = optuna.create_study(direction="minimize", pruner=pruner)
	study.optimize(objective, n_trials=TRIALS, timeout=None, show_progress_bar=True)

	print("Number of finished trials: {}".format(len(study.trials)))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: {}".format(trial.value))

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))

	df = study.trials_dataframe()
	df.to_csv('Results/VAE_optuna.csv')