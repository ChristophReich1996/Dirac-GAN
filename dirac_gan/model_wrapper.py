from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import HYPERPARAMETERS
from .loss import WassersteinGANLossGPDiscriminator, R1


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
        self.generator: nn.Module = generator
        self.discriminator: nn.Module = discriminator
        self.generator_optimizer: torch.optim.Optimizer = generator_optimizer
        self.discriminator_optimizer: torch.optim.Optimizer = discriminator_optimizer
        self.generator_loss_function: nn.Module = generator_loss_function
        self.discriminator_loss_function: nn.Module = discriminator_loss_function
        self.regularization_loss: nn.Module = regularization_loss

    def generate_trajectory(self, steps: int = 10, instance_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method generates gradient trajectory.
        :param steps: (int) Steps to utilize for each parameter in the range of [-2, 2]
        :param instance_noise: (bool) If true instance noise is utilized
        :return: (Tuple[torch.Tensor, torch.Tensor]) Parameters of the shape [steps^2, 2 (gen. dis.)] and parameter
        gradients of the shape [steps^2, 2 (gen. grad., dis. grad.)]
        """
        # Init list to store gradients
        gradients = []
        # Make parameters
        generator_parameters: torch.Tensor = torch.linspace(start=-2, end=2, steps=steps)
        discriminator_parameters: torch.Tensor = torch.linspace(start=-2, end=2, steps=steps)
        # Make parameter grid
        generator_parameters, discriminator_parameters = torch.meshgrid(generator_parameters, discriminator_parameters)
        generator_parameters = generator_parameters.reshape(-1)
        discriminator_parameters = discriminator_parameters.reshape(-1)
        # Iterate over parameter combinations
        for generator_parameter, discriminator_parameter in zip(generator_parameters, discriminator_parameters):
            # Set parameters
            self.generator.set_weight(generator_parameter)
            self.discriminator.set_weight(discriminator_parameter)
            ########## Generator gradient ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make fake prediction
            fake_prediction: torch.Tensor = self.discriminator(self.generator(get_noise(2048)))
            # Compute generator loss
            generator_loss: torch.Tensor = self.generator_loss_function(fake_prediction)
            # Compute gradients
            generator_loss.backward()
            # Save generator gradient
            generator_gradient: float = self.generator.get_gradient()
            ########## Discriminator gradient ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make real prediction
            real_samples: torch.Tensor = torch.zeros(HYPERPARAMETERS["batch_size"], 1)
            if instance_noise:
                real_samples: torch.Tensor = real_samples \
                                             + torch.randn(HYPERPARAMETERS["batch_size"], 1) \
                                             * HYPERPARAMETERS["in_scale"]
            real_samples.requires_grad = True
            real_prediction: torch.Tensor = self.discriminator(real_samples)
            # Make fake prediction
            noise: torch.Tensor = get_noise(1)
            fake: torch.Tensor = self.generator(noise)
            fake_prediction: torch.Tensor = self.discriminator(fake)
            # Compute generator loss
            if isinstance(self.discriminator_loss_function, WassersteinGANLossGPDiscriminator):
                discriminator_loss: torch.Tensor = self.discriminator_loss_function(real_prediction, fake_prediction,
                                                                                    self.discriminator,
                                                                                    torch.zeros(1, 1),
                                                                                    fake.detach())
            else:
                discriminator_loss: torch.Tensor = self.discriminator_loss_function(real_prediction, fake_prediction)
            # Compute gradient penalty if utilized
            if self.regularization_loss is not None:
                if isinstance(self.regularization_loss, R1):
                    discriminator_loss: torch.Tensor = discriminator_loss + self.regularization_loss(
                        real_prediction, real_samples)
                else:
                    discriminator_loss: torch.Tensor = discriminator_loss + self.regularization_loss(
                        fake_prediction, noise)
            # Compute gradients
            discriminator_loss.backward()
            # Save generator gradient
            discriminator_gradient: float = self.discriminator.get_gradient()
            # Save both gradients
            gradients.append((generator_gradient, discriminator_gradient))
        return torch.stack((generator_parameters, discriminator_parameters), dim=-1), torch.tensor(gradients)

    def train(self, instance_noise: bool = True) -> torch.Tensor:
        """
        Method trains the DiracGAN
        :param instance_noise: (bool) If true instance noise is utilized
        :param (torch.Tensor) History of generator and discriminator parameters [training iterations, 2 (gen., dis.)]
        """
        # Set initial weights
        self.generator.set_weight(torch.ones(1))
        self.discriminator.set_weight(torch.ones(1))
        # Init list to store the parameter history
        parameter_history = []
        # Perform training
        for iteration in range(HYPERPARAMETERS["training_iterations"]):
            ########## Generator training ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make fake prediction
            fake_prediction: torch.Tensor = self.discriminator(self.generator(get_noise(HYPERPARAMETERS["batch_size"])))
            # Compute generator loss
            generator_loss: torch.Tensor = self.generator_loss_function(fake_prediction)
            # Compute gradients
            generator_loss.backward()
            # Perform optimization
            self.generator_optimizer.step()
            ########## Disciminator training ###########
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            # Make real prediction
            real_samples: torch.Tensor = torch.zeros(HYPERPARAMETERS["batch_size"], 1)
            if instance_noise:
                real_samples: torch.Tensor = real_samples \
                                             + torch.randn(HYPERPARAMETERS["batch_size"], 1) \
                                             * HYPERPARAMETERS["in_scale"]
            real_samples.requires_grad = True
            real_prediction: torch.Tensor = self.discriminator(real_samples)
            # Make fake prediction
            noise: torch.Tensor = get_noise(2024)
            fake: torch.Tensor = self.generator(noise)
            fake_prediction: torch.Tensor = self.discriminator(fake)
            # Compute generator loss
            if isinstance(self.discriminator_loss_function, WassersteinGANLossGPDiscriminator):
                discriminator_loss: torch.Tensor = self.discriminator_loss_function(real_prediction, fake_prediction,
                                                                                    self.discriminator,
                                                                                    torch.zeros(2024, 1),
                                                                                    fake.detach())
            else:
                discriminator_loss: torch.Tensor = self.discriminator_loss_function(real_prediction, fake_prediction)
            # Compute gradient penalty if utilized
            if self.regularization_loss is not None:
                if isinstance(self.regularization_loss, R1):
                    discriminator_loss: torch.Tensor = discriminator_loss + self.regularization_loss(
                        real_prediction, real_samples)
                else:
                    discriminator_loss: torch.Tensor = discriminator_loss + self.regularization_loss(
                        fake_prediction, noise)
            # Compute gradients
            discriminator_loss.backward()
            # Perform optimization
            self.discriminator_optimizer.step()
            # Save parameters
            parameter_history.append((self.generator.get_weight(),
                                      self.discriminator.get_weight()))
        return torch.tensor(parameter_history)


def get_noise(batch_size: int) -> torch.Tensor:
    """
    Generates a noise tensor
    :param batch_size: (int) Batch size to be utilized
    :return: (torch.Tensor) Noise tensor
    """
    return 4. * torch.rand(batch_size, 1, requires_grad=True) - 1.
