import torch
import matplotlib.pyplot as plt

import dirac_gan

if __name__ == '__main__':
    # Init generator and discriminator
    generator = dirac_gan.Generator()
    discriminator = dirac_gan.Discriminator()
    # Init loss function
    generator_loss_function = dirac_gan.NSGANLossGenerator()
    discriminator_loss_function = dirac_gan.NSGANLossDiscriminator()
    # Init optimizers
    generator_optimizer = torch.optim.SGD(params=generator.parameters(), lr=dirac_gan.HYPERPARAMETERS["lr"],
                                          momentum=0.)
    discriminator_optimizer = torch.optim.SGD(params=discriminator.parameters(), lr=dirac_gan.HYPERPARAMETERS["lr"],
                                              momentum=0.)
    # Make model wrapper
    model_wrapper = dirac_gan.ModelWrapper(generator=generator,
                                           discriminator=discriminator,
                                           generator_optimizer=generator_optimizer,
                                           discriminator_optimizer=discriminator_optimizer,
                                           generator_loss_function=generator_loss_function,
                                           discriminator_loss_function=discriminator_loss_function)
    # Get trajectory
    parameters, gradients = model_wrapper.generate_trajectory()
    plt.quiver(parameters[..., 1], parameters[..., 0], -gradients[..., 1], -gradients[..., 0])
    # Perform training
    parameter_history = model_wrapper.train()
    # Plot parameter history
    print(parameter_history)
    plt.scatter(parameter_history[..., 1], parameter_history[..., 0])
    plt.grid()
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.xlabel("$\\theta$")
    plt.ylabel("$\\psi$")
    plt.show()
