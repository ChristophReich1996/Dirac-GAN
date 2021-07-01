import torch
import torch.nn as nn
import torch.nn.utils


class Generator(nn.Module):
    """
    Simple generator network of the DiracGAN with a single linear layer
    """

    def __init__(self, spectral_norm: bool = False) -> None:
        """
        Constructor method
        :param spectral_norm: (bool) If true spectral normalization is utilized
        """
        # Call super constructor
        super(Generator, self).__init__()
        # Init linear layer
        self.linear_layer = nn.Linear(in_features=1, out_features=1, bias=False)
        # Init weight of linear layer
        self.linear_layer.weight.data.fill_(1)
        # Apply spectral normalization if utilized
        if spectral_norm:
            self.linear_layer = torch.nn.utils.spectral_norm(self.linear_layer)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator
        :param noise: (torch.Tensor) Input noise tensor of the shape [batch size, 1]
        :return: (torch.Tensor) Generated samples of the shape [batch size, 1]
        """
        return self.linear_layer(noise)
