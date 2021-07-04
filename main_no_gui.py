import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QComboBox, QCheckBox
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import dirac_gan

if __name__ == '__main__':
    # Check if instance noise is utilized
    instance_noise: bool = False
    # Check which regularization is utilized
    regularization: str = "R2 gradient penalty"
    # Check which GAN loss is utilized
    gan_loss: str = "Standard GAN"
    # Init generator and discriminator
    generator: nn.Module = dirac_gan.Generator()
    discriminator: nn.Module = dirac_gan.Discriminator()
    # Init loss functions
    if gan_loss == "Standard GAN":
        generator_loss: nn.Module = dirac_gan.GANLossGenerator()
        discriminator_loss: nn.Module = dirac_gan.GANLossDiscriminator()
    elif gan_loss == "Non-saturating GAN":
        generator_loss: nn.Module = dirac_gan.NSGANLossGenerator()
        discriminator_loss: nn.Module = dirac_gan.NSGANLossDiscriminator()
    elif gan_loss == "Wasserstein GAN":
        generator_loss: nn.Module = dirac_gan.WassersteinGANLossGenerator()
        discriminator_loss: nn.Module = dirac_gan.WassersteinGANLossDiscriminator()
    elif gan_loss == "Wasserstein GAN GP":
        generator_loss: nn.Module = dirac_gan.WassersteinGANLossGPGenerator()
        discriminator_loss: nn.Module = dirac_gan.WassersteinGANLossGPDiscriminator()
    elif gan_loss == "Least squares GAN":
        generator_loss: nn.Module = dirac_gan.LSGANLossGenerator()
        discriminator_loss: nn.Module = dirac_gan.LSGANLossDiscriminator()
    else:
        generator_loss: nn.Module = dirac_gan.HingeGANLossGenerator()
        discriminator_loss: nn.Module = dirac_gan.HingeGANLossDiscriminator()
    # Init regularization loss
    if regularization == "None":
        regularization_loss: nn.Module = None
    elif regularization == "R1 gradient penalty":
        regularization_loss: nn.Module = dirac_gan.R1()
    else:
        regularization_loss: nn.Module = dirac_gan.R2()
    # Init optimizers
    generator_optimizer: torch.optim.Optimizer = torch.optim.SGD(params=generator.parameters(),
                                                                 lr=dirac_gan.HYPERPARAMETERS["lr"],
                                                                 momentum=0.)
    discriminator_optimizer: torch.optim.Optimizer = torch.optim.SGD(params=discriminator.parameters(),
                                                                     lr=dirac_gan.HYPERPARAMETERS["lr"],
                                                                     momentum=0.)
    # Make model wrapper
    model_wrapper = dirac_gan.ModelWrapper(generator=generator,
                                           discriminator=discriminator,
                                           generator_optimizer=generator_optimizer,
                                           discriminator_optimizer=discriminator_optimizer,
                                           generator_loss_function=generator_loss,
                                           discriminator_loss_function=discriminator_loss,
                                           regularization_loss=regularization_loss)
    # Get trajectory
    parameters, gradients = model_wrapper.generate_trajectory()
    # Perform training
    parameter_history: torch.Tensor = model_wrapper.train(instance_noise=instance_noise)
    # Perform plotting
    plt.quiver(parameters[..., 0], parameters[..., 1], -gradients[..., 0], -gradients[..., 1])
    plt.scatter(parameter_history[..., 0], parameter_history[..., 1])
    plt.grid()
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.xlabel("$\\theta$")
    plt.ylabel("$\\psi$")
    plt.show()