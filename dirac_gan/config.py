from typing import Dict, Any

import torch.nn as nn
import functools

GENERATOR_CONFIG: Dict[str, Any] = {
    "features": ((8, 8), (8, 16), (16, 32), (32, 2)),
    "activation": functools.partial(nn.LeakyReLU, 0.2, True)
}

DISCRIMINATOR_CONFIG: Dict[str, Any] = {
    "features": ((2, 8), (8, 16), (16, 32), (32, 1)),
    "activation": functools.partial(nn.LeakyReLU, 0.2, True)
}
