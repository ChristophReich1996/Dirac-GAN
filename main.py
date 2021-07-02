import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QComboBox, QCheckBox
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import dirac_gan


class App(QMainWindow):
    """
    GUI application class
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(App, self).__init__()
        # Init some attributes regarding the GUI
        self.left: int = 50
        self.top: int = 50
        self.title: str = "DiracGAN"
        self.width: int = 300
        self.height: int = 200
        # General setting
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Make list box to choose regularization
        self.instance_noise_box = QCheckBox("Instance noise", self)
        self.instance_noise_box.resize(300, 50)
        self.instance_noise_box.move(self.width - 300, self.height - 200)
        # Make list box to choose regularization
        self.regularization_box = QComboBox(self)
        self.regularization_box.addItem("None")
        self.regularization_box.addItem("R1 gradient penalty")
        self.regularization_box.addItem("R2 gradient penalty")
        self.regularization_box.resize(300, 50)
        self.regularization_box.move(self.width - 300, self.height - 150)
        # Make list box to choose GAN loss
        self.gan_list_box = QComboBox(self)
        self.gan_list_box.addItem("Standard GAN")
        self.gan_list_box.addItem("Non-saturating GAN")
        self.gan_list_box.addItem("Wasserstein GAN")
        self.gan_list_box.addItem("Wasserstein GAN GP")
        self.gan_list_box.addItem("Least squares GAN")
        self.gan_list_box.addItem("Hinge GAN")
        self.gan_list_box.resize(300, 50)
        self.gan_list_box.move(self.width - 300, self.height - 100)
        # Make run button
        run_button = QPushButton("button", self)
        run_button.setText("Run training")
        run_button.resize(300, 50)
        run_button.move(self.width - 300, self.height - 50)
        run_button.clicked.connect(self.__train)
        # Show everything
        self.show()

    def __train(self) -> None:
        """
        Method performs DiracGAN training and plots the training process
        """
        # Check if instance noise is utilized
        instance_noise: bool = self.instance_noise_box.isChecked()
        # Check which regularization is utilized
        regularization: str = self.regularization_box.currentText()
        # Check which GAN loss is utilized
        gan_loss: str = self.gan_list_box.currentText()
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
        elif regularization == "R1":
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


if __name__ == '__main__':
    application = QApplication(sys.argv)
    _ = App()
    sys.exit(application.exec_())
