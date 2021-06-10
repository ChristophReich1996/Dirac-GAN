# DiracGAN

This repository implements (PyTorch) the DiracGAN proposed in the paper "Which Training Methods for GANs do actually Converge?" by [Mescheder](https://github.com/LMescheder) et al. [1]. The original implementation of the authors can be found [here](https://github.com/LMescheder/GAN_stability).

This work was done as part of the lecture deep generative models at TU Darmstadt supervised by [Dr. Anirban Mukhopadhyay](https://www.informatik.tu-darmstadt.de/gris/startseite_1/team/team_details_60224.en.jsp).

## Dependencies

DiracGAN is written in [PyTorch 1.8.1](https://pytorch.org/). No GPU is required but can be used to speed up computation. All additional dependencies can be seen in the [`requirements.txt`](requirements.txt) file. To install all dependencies simply run:

```shellscript
pip install -r requirements.txt
```

## Usage

## Rerefences

```bibtex
[1] @inproceedings{Mescheder2018,
    title={Which training methods for GANs do actually converge?},
    author={Mescheder, Lars and Geiger, Andreas and Nowozin, Sebastian},
    booktitle={International conference on machine learning},
    pages={3481--3490},
    year={2018},
    organization={PMLR}
}
```
