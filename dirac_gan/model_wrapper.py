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
                 discriminator_optimizer: torch.optim.Optimizer,
                 generator_loss_function: nn.Module,
                 discriminator_loss_function: nn.Module,
                 regularization_loss: Optional[nn.Module] = None,
                 device: Union[str, torch.device] = "cpu") -> None:
        """
        Constructor method
        :param generator: (nn.Module) Generator network
        :param discriminator: (nn.Module) Discriminator network
        :param generator_optimizer: (torch.optim.Optimizer) Generator optimizer
        :param discriminator_optimizer: (torch.optim.Optimizer) Discriminator optimizer
        :param generator_loss_function: (nn.Module) Generator loss function
        :param discriminator_loss_function: (nn.Module) Discriminator loss function
        :param regularization_loss: (Optional[nn.Module]) Regularization loss
        :param device: (str) Device to be utilized
        """
        # Save parameters
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_function = generator_loss_function
        self.discriminator_loss_function = discriminator_loss_function
        self.regularization_loss = regularization_loss
        self.device = device

    def train(self) -> torch.Tensor:
        """
        Method trains the DiracGAN
        :param (torch.Tensor) History of generator and discriminator parameters
        """
        # Init list to store the parameter history
        parameter_history = []
        # Perform training
        for iteration in range(HYPERPARAMETERS["training_iterations"]):
            print(iteration)
            ########## Generator training ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make fake prediction
            fake_prediction = self.discriminator(self.generator(torch.rand(HYPERPARAMETERS["batch_size"], 1)))
            # Compute generator loss
            generator_loss = self.generator_loss_function(fake_prediction)
            # Compute gradients
            generator_loss.backward()
            # Perform optimization
            self.generator_optimizer.step()
            ########## Generator training ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make real prediction
            real_prediction = self.discriminator(torch.zeros(HYPERPARAMETERS["batch_size"], 1))
            # Make fake prediction
            fake_prediction = self.discriminator(self.generator(torch.rand(HYPERPARAMETERS["batch_size"], 1)))
            # Compute generator loss
            discriminator_loss = self.discriminator_loss_function(real_prediction, fake_prediction)
            # Compute gradients
            discriminator_loss.backward()
            # Perform optimization
            self.discriminator_optimizer.step()
            # Save parameters
            parameter_history.append((self.generator.linear_layer.weight.data.item(),
                                      self.discriminator.linear_layer.weight.data.item()))
        return torch.tensor(parameter_history)
