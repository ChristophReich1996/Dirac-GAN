from typing import Optional, Union, Tuple

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
                 regularization_loss: Optional[nn.Module] = None) -> None:
        """
        Constructor method
        :param generator: (nn.Module) Generator network
        :param discriminator: (nn.Module) Discriminator network
        :param generator_optimizer: (torch.optim.Optimizer) Generator optimizer
        :param discriminator_optimizer: (torch.optim.Optimizer) Discriminator optimizer
        :param generator_loss_function: (nn.Module) Generator loss function
        :param discriminator_loss_function: (nn.Module) Discriminator loss function
        :param regularization_loss: (Optional[nn.Module]) Regularization loss
        """
        # Save parameters
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_function = generator_loss_function
        self.discriminator_loss_function = discriminator_loss_function
        self.regularization_loss = regularization_loss

    def generate_trajectory(self, steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method generates gradient trajectory.
        :return: (Tuple[torch.Tensor, torch.Tensor]) Parameters of the shape [steps^2, 2 (gen. dis.)] and parameter
        gradients of the shape [steps^2, 2 (gen. grad., dis. grad.)]
        """
        # Init list to store gradients
        gradients = []
        # Make parameters
        generator_parameters = torch.linspace(start=-2, end=2, steps=steps)
        discriminator_parameters = torch.linspace(start=-2, end=2, steps=steps)
        # Make parameter grid
        generator_parameters, discriminator_parameters = torch.meshgrid(generator_parameters, discriminator_parameters)
        generator_parameters = generator_parameters.reshape(-1)
        discriminator_parameters = discriminator_parameters.reshape(-1)
        # Iterate over parameter combinations
        for generator_parameter, discriminator_parameter in zip(generator_parameters, discriminator_parameters):
            # Set parameters
            self.generator.set_weight(generator_parameter)
            self.discriminator.set_weight(discriminator_parameter)
            ########## Generator training ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make fake prediction
            fake_prediction = self.discriminator(self.generator(get_noise(2048)))
            # Compute generator loss
            generator_loss = self.generator_loss_function(fake_prediction)
            # Compute gradients
            generator_loss.backward()
            # Save generator gradient
            generator_gradient = self.generator.get_gradient()
            ########## Generator training ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make real prediction
            real_prediction = self.discriminator(torch.zeros(2024, 1))
            # Make fake prediction
            fake_prediction = self.discriminator(self.generator(get_noise(2024)))
            # Compute generator loss
            discriminator_loss = self.discriminator_loss_function(real_prediction, fake_prediction)
            # Compute gradients
            discriminator_loss.backward()
            # Save generator gradient
            discriminator_gradient = self.discriminator.get_gradient()
            # Save both gradients
            gradients.append((generator_gradient, discriminator_gradient))
        return torch.stack((generator_parameters, discriminator_parameters), dim=-1), torch.tensor(gradients)

    def train(self) -> torch.Tensor:
        """
        Method trains the DiracGAN
        :param (torch.Tensor) History of generator and discriminator parameters [training iterations, 2 (gen., dis.)]
        """
        # Set initial weights
        self.generator.set_weight(torch.ones(1))
        self.discriminator.set_weight(torch.ones(1))
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
            fake_prediction = self.discriminator(self.generator(get_noise(HYPERPARAMETERS["batch_size"])))
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
            fake_prediction = self.discriminator(self.generator(get_noise(HYPERPARAMETERS["batch_size"])))
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


def get_noise(batch_size: int) -> torch.Tensor:
    """
    Generates a noise tensor
    :param batch_size: (int) Batch size to be utilized
    :return: (torch.Tensor) Noise tensor
    """
    return 4. * torch.rand(batch_size, 1) - 1.
