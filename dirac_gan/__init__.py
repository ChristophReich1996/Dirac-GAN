# Import generator and discriminator
from dirac_gan.generator import get_generator as Generator
from dirac_gan.discriminator import get_discriminator as Discriminator
# Import losses
from dirac_gan.loss import GANLossGenerator, GANLossDiscriminator, NSGANLossGenerator, NSGANLossDiscriminator, \
    LSGANLossGenerator, LSGANLossDiscriminator, WassersteinGANLossGenerator, WassersteinGANLossDiscriminator, \
    WassersteinGANLossGPGenerator, WassersteinGANLossGPDiscriminator, HingeGANLossGenerator, HingeGANLossDiscriminator, \
    R1, R2
