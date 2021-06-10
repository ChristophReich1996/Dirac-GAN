from typing import Dict, Any, Type, Tuple

import torch
import torch.nn as nn

from dirac_gan.config import GENERATOR_CONFIG


class Generator(nn.Module):
    """
    This class implements a simple residual generator similar to:
    https://arxiv.org/pdf/1801.04406.pdf (uses no residual skip connections)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Constructor method
        :param config: (Dict[str, Any]) Config dict including all hyperparameters
        """
        # Call super constructor
        super(Generator, self).__init__()
        # Get parameter
        features: Tuple[Tuple[int, int], ...] = config["features"]
        activation: Type[nn.Module] = config["activation"]
        spectral_norm: bool = config["spectral_norm"]
        # Init blocks
        self.blocks = nn.Sequential(
            *[ResNetBlock(in_features=feature[0], out_features=feature[1], activation=activation,
                          spectral_norm=spectral_norm) for feature in features]
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator
        :param noise: (torch.Tensor) Input noise tensor of the shape [batch size, n samples]
        :return: (torch.Tensor) Generated samples of the shape [batch size, 2]
        """
        return self.blocks(noise)


class ResNetBlock(nn.Module):
    """
    This class implements a simple residual feed forward block with two linear layers.
    """

    def __init__(self, in_features: int, out_features: int, spectral_norm: bool, activation: Type[nn.Module]) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input channels
        :param out_features: (int) Number of output channels
        :param spectral_norm: (bool) If true spectral norm is utilized
        """
        # Call super constructor
        super(ResNetBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            get_linear_layer(in_features=in_features, out_features=out_features, spectral_norm=spectral_norm),
            activation(),
            get_linear_layer(in_features=out_features, out_features=out_features, spectral_norm=spectral_norm),
            activation(),
        )
        # Init skip connection
        self.skip_connection = get_linear_layer(
            in_features=in_features, out_features=out_features,
            spectral_norm=spectral_norm) if in_features != out_features else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in features]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out features]
        """
        return self.main_mapping(input) + self.skip_connection(input)


def get_linear_layer(in_features: int, out_features: int, spectral_norm: bool) -> nn.Module:
    """
    Initializes a linear layer with spectral normalization if utilized
    :param in_features: (int) Number of input channels
    :param out_features: (int) Number of output channels
    :param spectral_norm: (bool) If true spectral norm is utilized
    :return: (nn.Module) Linear layer
    """
    if spectral_norm:
        return nn.utils.spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=False))
    return nn.Linear(in_features=in_features, out_features=out_features, bias=False)


def get_generator(spectral_norm: bool = False) -> nn.Module:
    """
    Model builds the generator network
    :param spectral_norm: (bool) If true spectral norm is utilized
    :return: (nn.Module) Generator network
    """
    # Get config
    config = GENERATOR_CONFIG
    # Set spectral norm usage
    config["spectral_norm"] = spectral_norm
    # Init model
    return Generator(config=config)
