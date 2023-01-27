import torch
from torch import nn
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from abc import abstractmethod
# from torch import torch.tensor as torch.tensor

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: torch.tensor) -> List[torch.tensor]:
        raise NotImplementedError

    def decode(self, input: torch.tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> torch.tensor:
        raise NotImplementedError

    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.tensor:
        pass


class DFCVAE(BaseVAE):

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 latent_dim: int,
                 hidden_dims: List = None,
                 alpha:float = 1,
                 beta:float = 0.5,
                 **kwargs) -> None:
        super(DFCVAE, self).__init__()

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta

        in_channels = input_shape[0]

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, 
                              stride= 1, 
                              padding  = 0),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.test_out = self.encoder(torch.zeros(1, *input_shape))
        self.shape_test_out = self.test_out.shape
        size_test_out = self.test_out.flatten().shape[0]

        self.fc_mu = nn.Linear(size_test_out, latent_dim)
        self.fc_var = nn.Linear(size_test_out, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, size_test_out)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 1,
                                       padding=0,
                                       output_padding=0),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               input_shape[0],
                                               kernel_size=3,
                                               stride=1,
                                               padding=0,
                                               output_padding=0),
                            #nn.BatchNorm2d(hidden_dims[-1]),
                            #nn.LeakyReLU(),
                            #nn.Conv2d(hidden_dims[-1], out_channels= 3,
                            #          kernel_size= 3, padding= 1),
                            nn.Tanh())

        self.feature_network = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)

        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.feature_network.eval()


    def encode(self, input: torch.tensor) -> List[torch.tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.tensor) Input torch.tensor to encoder [N x C x H x W]
        :return: (torch.tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.tensor) -> torch.tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.tensor) [B x D]
        :return: (torch.tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.shape_test_out[1], self.shape_test_out[2], self.shape_test_out[3])
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (torch.tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (torch.tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (torch.tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.tensor, **kwargs) -> List[torch.tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)

        # Transform single channel images to 3 channels by repeating the channel
        if recons.shape[1] == 1:
            recons_features = self.extract_features(recons.repeat(1, 3, 1, 1))
            input_features = self.extract_features(input.repeat(1, 3, 1, 1))

        else:
            recons_features = self.extract_features(recons)
            input_features = self.extract_features(input)

        return  [recons, input, recons_features, input_features, mu, log_var]

    def extract_features(self,
                         input: torch.tensor,
                         feature_layers: List = None) -> List[torch.tensor]:
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (torch.tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features

    def loss_function(self, mu, log_var, input, out, gt_imgs, sensing_masks, recons_features, input_features):

        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        reconstruction_loss = torch.mean((1.0 - sensing_masks) * torch.abs(out - gt_imgs)) + torch.mean(sensing_masks * torch.abs(out - input))


        feature_loss = 0.0
        for (r, i) in zip(recons_features, input_features):
            feature_loss += F.mse_loss(r, i)

        kl_divergence = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        cross_entropy = F.binary_cross_entropy(out, input, reduction='mean')

        perceptual_loss = torch.mean(torch.abs(out - gt_imgs))

        loss =  reconstruction_loss + feature_loss + kl_divergence + cross_entropy + perceptual_loss

        return loss, cross_entropy, kl_divergence, perceptual_loss, reconstruction_loss, feature_loss

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.tensor) [B x C x H x W]
        :return: (torch.tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


if __name__ == '__main__':

    # Test the DFCVAE using a random input. Generate a random image and obtain the output.
    # Represent both input and output one next to the other.
    import matplotlib.pyplot as plt
    
    # Create a random image
    img = torch.randn(1, 1, 64, 32)
    # Create a DFCVAE model
    model = DFCVAE(input_shape=(1, 64, 32),
                 latent_dim = 256)

    # Obtain the output
    output = model(img)
    # Plot both images 
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img[0].permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(output[0][0].detach().permute(1, 2, 0))
    plt.show()




