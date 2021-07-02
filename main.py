import torch
import matplotlib.pyplot as plt

import dirac_gan

if __name__ == '__main__':
    # Set losses
    losses = [(dirac_gan.GANLossGenerator, dirac_gan.GANLossDiscriminator),
              (dirac_gan.NSGANLossGenerator, dirac_gan.NSGANLossDiscriminator),
              (dirac_gan.WassersteinGANLossGenerator, dirac_gan.WassersteinGANLossDiscriminator),
              (dirac_gan.WassersteinGANLossGPGenerator, dirac_gan.WassersteinGANLossGPDiscriminator),
              (dirac_gan.LSGANLossGenerator, dirac_gan.LSGANLossDiscriminator),
              (dirac_gan.HingeGANLossGenerator, dirac_gan.HingeGANLossDiscriminator)]
    loss_names = ["standard_gan", "non_saturating_gan", "wasserstein_gan", "wasserstein_gp_gan", "ls_gan", "hinge_gan"]
    for loss, loss_name in zip(losses, loss_names):
        # Init generator and discriminator
        generator = dirac_gan.Generator()
        discriminator = dirac_gan.Discriminator()
        # Init loss function
        generator_loss_function = loss[0]()
        discriminator_loss_function = loss[1]()
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
                                               discriminator_loss_function=discriminator_loss_function,
                                               regularization_loss=dirac_gan.R2())
        # Get trajectory
        parameters, gradients = model_wrapper.generate_trajectory()
        plt.quiver(parameters[..., 0], parameters[..., 1], -gradients[..., 0], -gradients[..., 1])
        # Perform training
        parameter_history = model_wrapper.train()
        # Plot parameter history
        plt.scatter(parameter_history[..., 0], parameter_history[..., 1])
        plt.grid()
        plt.xlim((-2, 2))
        plt.ylim((-2, 2))
        plt.xlabel("$\\theta$")
        plt.ylabel("$\\psi$")
        plt.savefig(f"{loss_name}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
