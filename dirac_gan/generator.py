import torch
import torch.nn as nn
import torch.nn.utils


class Generator(nn.Module):
    """
    Simple generator network of the DiracGAN with a single linear layer
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(Generator, self).__init__()
        # Init linear layer
        self.parameter = nn.Parameter(torch.ones(1, 1))

    def get_weight(self) -> float:
        """
        Method returns the weight of the generator
        :return: (float) Generator weight
        """
        return self.parameter.data.item()

    def get_gradient(self) -> float:
        """
        Method returns the gradient of the weight
        :return: (float) Gradient value
        """
        return self.parameter.grad.item()

    def set_weight(self, weight: torch.Tensor) -> None:
        """
        Method sets the weight factor of the linear layer
        :param weight: (torch.Tensor) Value to be set
        """
        # Reshape given tensor
        weight = weight.view(1, 1)
        # Check size of parameter
        self.parameter.data = weight

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator
        :param noise: (torch.Tensor) Input noise tensor of the shape [batch size, 1]
        :return: (torch.Tensor) Generated samples of the shape [batch size, 1]
        """
        prediction = torch.repeat_interleave(self.parameter, repeats=noise.shape[0], dim=0)
        return prediction
