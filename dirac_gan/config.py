from typing import Dict, Any

HYPERPARAMETERS: Dict[str, Any] = {
    "training_iterations": 500,
    "batch_size": 1,
    "lr": .1,
    "in_scale": 0.4,
    "r1_w": 0.2,
    "r2_w": 0.2,
    "gp_w": 0.25,
    "dra_w": 0.1,
    "rlc_af": 1.,
    "rlc_ar": 1.,
    "rlc_w": 0.2
}
