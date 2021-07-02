# DiracGAN
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ChristophReich1996/Cell-DETR/blob/master/LICENSE)

<table>
  <tr>
    <td> Standard GAN loss </td>
    <td> Non-saturating GAN loss </td>
    <td> Wasserstein GAN </td>
  </tr> 
  <tr>
    <td> <img src="/images/standard_gan.png"  alt="1" width = 200px height = 150px ></td>
    <td> <img src="/images/non_saturating_gan.png" alt="2" width = 200px height = 150px></td>
    <td> <img src="/images/wasserstein_gan.png"  alt="3" width = 200px height = 150px ></td>
  </tr> 
  <tr>
    <td> Wasserstein GAN loss + GP </td>
    <td> Least squares GAN </td>
    <td> Hinge GAN </td>
  </tr> 
  <tr>
    <td> <img src="/images/wasserstein_gp_gan.png"  alt="5" width = 200px height = 150px ></td>
    <td> <img src="/images/ls_gan.png" alt="6" width = 200px height = 150px></td>
    <td> <img src="/images/hinge_gan.png"  alt="7" width = 200px height = 150px ></td>
  </tr>
</table>

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
