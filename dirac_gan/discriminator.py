from typing import Dict, Any, Type, Tuple

import torch
import torch.nn as nn

from dirac_gan.generator import Generator
from dirac_gan.config import DISCRIMINATOR_CONFIG


class Discriminator(Generator):
    """
    This class implements a simple residual discriminator similar to:
    https://arxiv.org/pdf/1801.04406.pdf (uses no residual skip connections)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Constructor method
        :param config: (Dict[str, Any]) Config dict including all hyperparameters
        """
        # Call super constructor
        super(Discriminator, self).__init__(config=config)


def get_discriminator(spectral_norm: bool = False) -> nn.Module:
    """
    Model builds the discriminator network
    :param spectral_norm: (bool) If true spectral norm is utilized
    :return: (nn.Module) Discriminator network
    """
    # Get config
    config = DISCRIMINATOR_CONFIG
    # Set spectral norm usage
    config["spectral_norm"] = spectral_norm
    # Init model
    return Generator(config=config)
