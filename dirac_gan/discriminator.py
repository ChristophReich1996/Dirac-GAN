from dirac_gan.generator import Generator


class Discriminator(Generator):
    """
    Simple discriminator network of the DiracGAN with a single linear layer.
    """

    def __init__(self, spectral_norm: bool = False) -> None:
        """
        Constructor method
        :param spectral_norm: (bool) If true spectral normalization is utilized
        """
        # Call super constructor
        super(Discriminator, self).__init__(spectral_norm=spectral_norm)
