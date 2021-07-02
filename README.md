# DiracGAN
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ChristophReich1996/Cell-DETR/blob/master/LICENSE)

This repository implements (PyTorch) the DiracGAN proposed in the paper "Which Training Methods for GANs do actually Converge?" by [Mescheder](https://github.com/LMescheder) et al. [1]. The original implementation of the authors can be found [here](https://github.com/LMescheder/GAN_stability).

This work was done as part of the lecture deep generative models at TU Darmstadt supervised by [Dr. Anirban Mukhopadhyay](https://www.informatik.tu-darmstadt.de/gris/startseite_1/team/team_details_60224.en.jsp).

**Parts of this implementation are taken from my recent [mode collapse example repository](https://github.com/ChristophReich1996/Mode_Collapse).**

## Dependencies

DiracGAN is written in [PyTorch 1.8.1](https://pytorch.org/). No GPU is required! All additional dependencies can be seen in the [`requirements.txt`](requirements.txt) file. To install all dependencies simply run:

```shellscript
pip install -r requirements.txt
```

[Older version of PyTorch](https://pytorch.org/get-started/previous-versions/) may also allows running the code without issues.

## Usage

The implementation provides a simple GUI to run all DiracGAN experiment with different settings. Simply run:

```shell script
python main.py
```

Set the desired parameters in the GUI and click on run to perform training. This could take a few seconds. If the training
is finished all results are plotted and shown.

## References

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
