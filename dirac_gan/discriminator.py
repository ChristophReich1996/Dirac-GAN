import torch
import torch.nn as nn
import torch.nn.utils


class Discriminator(nn.Module):
    """
    Simple discriminator network of the DiracGAN with a single linear layer
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(Discriminator, self).__init__()
        # Init linear layer
        self.linear_layer = nn.Linear(in_features=1, out_features=1, bias=False)
        # Init weight of linear layer
        self.linear_layer.weight.data.fill_(1)

    def get_weight(self) -> float:
        """
        Method returns the weight of the discriminator
        :return: (float) Discriminator weight
        """
        return self.linear_layer.weight.data.item()

    def get_gradient(self) -> float:
        """
        Method returns the gradient of the weight
        :return: (float) Gradient value
        """
        return self.linear_layer.weight.grad.item()

    def set_weight(self, weight: torch.Tensor) -> None:
        """
        Method sets the weight factor of the linear layer
        :param weight: (torch.Tensor) Value to be set
        """
        # Reshape given tensor
        weight = weight.view(1, 1)
        # Check size of parameter
        self.linear_layer.weight.data = weight

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator
        :param noise: (torch.Tensor) Input noise tensor of the shape [batch size, 1]
        :return: (torch.Tensor) Generated samples of the shape [batch size, 1]
        """
        return self.linear_layer(noise)
