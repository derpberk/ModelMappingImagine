
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import vgg19_bn, VGG19_BN_Weights
import numpy as np

"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
	def __init__(self, input_shape = (1, 64, 64), zDim=256, deep_feature_coherent = False, **kwargs):
		super(VAE, self).__init__()


		# Whether to use the deep feature coherent loss
		self.dfc = deep_feature_coherent

		# Set the hyperparameters for weighting tyhe loss
		if self.dfc:
			self.L_dfc = kwargs['L_dfc']

		self.L_kl = kwargs['L_kl']
		self.L_p  = kwargs['L_p']
		self.L_r  = kwargs['L_r']


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

		# Initialize the vgg19_bn model for deep feature coherent loss
		if self.dfc:
			self.feature_extractor = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)

	def set_optimizer(self):
		# Set optimizer 
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)


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

	def extract_features(self,
						input: torch.tensor,
						feature_layers = None):
		"""
		Extracts the features from the pretrained model
		at the layers indicated by feature_layers.
		:param input: (torch.tensor) [B x C x H x W]
		:param feature_layers: List of string of IDs
		:return: List of the extracted features
		"""

		# Transform single channel images to 3 channels by repeating the channel
		if input.shape[1] == 1:
			input = input.repeat(1, 3, 1, 1)

		if feature_layers is None:
			feature_layers = ['14', '24', '34', '43']

		features = []
		result = input
		for (key, module) in self.feature_extractor.features._modules.items():
			result = module(result)
			if(key in feature_layers):
				features.append(result)

		return features

	def forward(self, x):

		# The entire pipeline of the VAE: encoder -> reparameterization -> decoder
		# output, mu, and logVar are returned for loss computation
		mu, logVar = self.encode(x)
		z = self.reparameterize(mu, logVar)
		out = self.decode(z)

		# Compute the deep feature coherent loss if self.dfc is True
		if self.dfc and self.training:

			# Extract features of the input 
			features_input = self.extract_features(x)
			# Extract features of the output
			features_outut = self.extract_features(out)

			return out, mu, logVar, features_input, features_outut

		else:

			return out, mu, logVar

	def compute_loss(self, mu, logVar, input, out, gts, sensing_masks, features_input = None, features_output = None):

		# Compute the reconstruction loss and KL divergence loss
		# The loss is the summation of the two losses
		# The loss is used for backpropagation
		# The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt

		# Check if the deep feature coherent loss is used and features_input and features_output are not None
		features_loss = 0

		if self.dfc:
			assert not (features_input is None or features_output is None), "The deep feature coherent loss is used but features_input or features_output is None"
			# Compute the deep feature coherent loss
			for (r, i) in zip(features_output, features_input):
				features_loss += F.mse_loss(r, i)

		kl_divergence = 0.5 * torch.mean(-1 - logVar + mu.pow(2) + logVar.exp())
		perceptual_loss = torch.mean(torch.abs(out - gts))
		reconstruction_loss = torch.mean((1.0 - sensing_masks) * torch.abs(out - gts)) + torch.mean(sensing_masks * torch.abs(out - input))

		loss = self.L_kl * kl_divergence + self.L_p * perceptual_loss + self.L_r * reconstruction_loss + self.L_dfc * features_loss

		return loss, kl_divergence, perceptual_loss, reconstruction_loss, features_loss

	def imagine(self, input_image, N = 1):
		
		# The imagine function takes in the input image and generate the output image
		# The input image is fed into the encoder to generate the latent representation
		# The latent representation is fed into the decoder to generate the output image
		# The output image is returned

		# Encode the input image

		input_image = torch.as_tensor(input_image, dtype=torch.float32).to('cuda:0')
		mu, logVar = self.encode(input_image)

		if N == 0:
			# Sample from the mean of the distribution #
			z = self.reparameterize(mu, torch.zeros_like(logVar))
			# Decode the latent representation
			out = self.decode(z).squeeze(0)
			return out, None
		else:
			# Reparameterize the latent representation N times: call self.reparameterize N times and stack the results in the first dimension #
			z = torch.stack([self.reparameterize(mu, logVar) for _ in range(N)], dim=0)
			# Decode the latent representation
			out = self.decode(z)
			# Average the results along the first dimension
			mean_out = torch.mean(out, dim=0).squeeze(0)
			# Obtain the standard deviation along the first dimension
			std_out = torch.std(out, dim=0).squeeze(0)
			# Return 
			return mean_out, std_out

	def evaluate_batch(self, batch, N = 10):
		""" Evaluate a batch of images using the mse """

		with torch.no_grad():

				gt_imgs = batch[:, 0, :, :].unsqueeze(1).to('cuda:0')
				sensing_masks = batch[:, 1, :, :].unsqueeze(1).to('cuda:0')

				# The input images are the first channel multiplied by the sensing mask
				imgs = gt_imgs * sensing_masks

		# Forward pass
		mean_out, _ = self.imagine(imgs, N = N)

		# Compute the mse between the unmasked mean_out and the unmasked ground truth
		return F.mse_loss(mean_out * (1.0 - sensing_masks),  gt_imgs * (1.0 - sensing_masks)).item()

	def train_batch(self, batch):
		""" Train a batch of images using the mse """

		with torch.no_grad():

				gt_imgs = batch[:, 0, :, :].unsqueeze(1).to('cuda:0')
				sensing_masks = batch[:, 1, :, :].unsqueeze(1).to('cuda:0')

				# The input images are the first channel multiplied by the sensing mask
				imgs = gt_imgs * sensing_masks

		# Feeding a batch of images into the network to obtain the output image, mu, and logVar
		out, mu, logVar, fea_in, fea_out = self.forward(imgs)

		# Compute the loss
		loss, cross_entropy, kl_divergence, perceptual_loss, reconstruction_loss  = self.compute_loss(mu, logVar, imgs, out, gt_imgs, sensing_masks, fea_in, fea_out)

		# Backpropagation based on the loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()




if __name__ == '__main__':
	# Test the VAE

	# load matrix of the map 
	import numpy as np

	map = np.genfromtxt('Maps/example_map.csv')

	input_shape = (1, *map.shape)
	model = VAE(input_shape = input_shape, zDim=256, L_kl = 1.0, L_r = 1.0, L_p = 1.0, L_dfc=1.0, deep_feature_coherent=True)
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



	