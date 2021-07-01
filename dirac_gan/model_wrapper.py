from typing import Optional, Union

import torch
import torch.nn as nn

from .config import HYPERPARAMETERS


class ModelWrapper(object):
    """
    This class implements a wrapper for the DiracGAN.
    """

    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 generator_optimizer: torch.optim.Optimizer,
                 discriminator_optimizer: torch.optim.optimizer,
                 loss_function: nn.Module,
                 regularization_loss: Optional[nn.Module] = None,
                 device: Union[str, torch.device] = "cpu") -> None:
        """
        Constructor method
        :param generator: (nn.Module) Generator network
        :param discriminator: (nn.Module) Discriminator network
        :param generator_optimizer: (torch.optim.Optimizer) Generator optimizer
        :param discriminator_optimizer: (torch.optim.Optimizer) Discriminator optimizer
        :param loss_function: (nn.Module) Loss function
        :param regularization_loss: (Optional[nn.Module]) Regularization loss
        :param device: (str) Device to be utilized
        """
        # Save parameters
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_function = loss_function
        self.regularization_loss = regularization_loss
        self.device = device
