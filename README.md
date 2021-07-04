# Dirac-GAN
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ChristophReich1996/Cell-DETR/blob/master/LICENSE)

This repository implements (PyTorch) the Dirac-GAN proposed in the paper "Which Training Methods for GANs do actually Converge?" by [Mescheder](https://github.com/LMescheder) et al. [1]. The original implementation of the authors can be found [here](https://github.com/LMescheder/GAN_stability).

This work was done as part of the lecture Deep Generative Models at TU Darmstadt held by [Dr. Anirban Mukhopadhyay](https://www.informatik.tu-darmstadt.de/gris/startseite_1/team/team_details_60224.en.jsp).

**Parts of this implementation are taken from my recent [mode collapse example repository](https://github.com/ChristophReich1996/Mode_Collapse).**

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
  <tr>
    <td> DRAGAN loss </td>
  </tr> 
  <tr>
    <td> <img src="/images/dra_gan.png"  alt="5" width = 200px height = 150px ></td>
  </tr>
</table>

This repository implements the following GAN losses and regularizers.

| Method | Generator loss | Discriminator loss |
| :--- | :--- | :--- |
| Original GAN loss | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}^{\text{GAN}}_{D}=-\mathbb{E}_{x\sim p_{d}}[\log(D(x))] - \mathbb{E}_{\hat{x}\sim p_{g}}[\log(1 - D(\hat{x}))]"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{G}^{\text{GAN}}=\mathbb{E}_{\hat{x}\sim p_{g}}[\log(1 - D(\hat{x}))]"> |
| Non-saturating GAN loss | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{D}^{\text{NSGAN}}=-\mathbb{E}_{x\sim p_{d}}[\log(D(x))] - \mathbb{E}_{\hat{x}\sim p_{g}}[\log(1 - D(\hat{x}))]"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{G}^{\text{NSGAN}}=-\mathbb{E}_{\hat{x}\sim p_{g}}[\log(D(\hat{x}))]"> |
| Wasserstein GAN loss | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{D}^{\text{WGAN}}=-\mathbb{E}_{x\sim p_{d}}[D(x)] %2B \mathbb{E}_{\hat{x}\sim p_{g}}[D(\hat{x})]"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{G}^{\text{WGAN}}=-\mathbb{E}_{\hat{x}\sim p_{g}}[D(\hat{x})]"> |
| Wasserstein GAN loss + grad. pen. | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{D}^{\text{WGANGP}}=\mathcal{L}_{D}^{\text{WGAN}} %2B \lambda\mathbb{E}_{\hat{x}\sim p_{g}}[(\lvert\lvert\nabla D(\alpha x %2B (1 - \alpha \hat{x}))\rvert\rvert_{2} - 1)^2]"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{G}^{\text{WGANGP}}=\mathcal{L}_{G}^{\text{WGAN}}"> |
| Least squares GAN loss | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}^{\text{LSGAN}}_{D}=-\mathbb{E}_{x\sim p_{d}}[(D(x) - 1)^2] %2B \mathbb{E}_{\hat{x}\sim p_{g}}[D(\hat{x})^2]"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}^{\text{LSGAN}}_{G}=-\mathbb{E}_{\hat{x}\sim p_{g}}[(D(\hat{x} - 1))^2]"> |
| Hinge GAN | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}^{\text{LSGAN}}_{D}=-\mathbb{E}_{x\sim p_{d}}[\min(0, D(x)-1] - \mathbb{E}_{\hat{x}\sim p_{g}}[\min(0, -D(\hat{x})-1)]"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}^{\text{LSGAN}}_{G}=\mathcal{L}^{\text{WGAN}}_{G}"> |
| DRAGAN | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{D}^{\text{DRAGAN}}=\mathcal{L}_{D}^{\text{GAN}} %2B \lambda\mathbb{E}_{\hat{x}\sim p_{d} %2B \mathcal{N}(0, c)}[(\lvert\lvert\nabla D(\hat{x})\rvert\rvert_{2} - 1)^2]"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{G}^{\text{DRAGAN}}=\mathcal{L}_{G}^{\text{GAN}}"> |

| Method | Generator loss |
| :--- | :--- |
| <img src="https://render.githubusercontent.com/render/math?math=R_{1}"> regularization | <img src="https://render.githubusercontent.com/render/math?math=R_{1}=\frac{\gamma}{2}\mathbb{E}_{x\sim p_{d}}[\lvert\lvert\nabla D(x)\rvert\rvert^2]"> |
| <img src="https://render.githubusercontent.com/render/math?math=R_{2}"> regularization | <img src="https://render.githubusercontent.com/render/math?math=R_{2}=\frac{\gamma}{2}\mathbb{E}_{\hat{x}\sim p_{g}}[\lvert\lvert\nabla D(x)\rvert\rvert^2]"> |
| <img src="https://render.githubusercontent.com/render/math?math=R_{\text{LC}}"> regularization | <img src="https://render.githubusercontent.com/render/math?math=R_{\text{LC}}=\mathbb{E}_{x\sim p_{d}}[\lvert\lvert D(x) - \alpha_{F}\rvert\rvert^{2}] %2B \mathbb{E}_{\hat{x}\sim p_{g}}[\lvert\lvert D(G(\hat{x})) - \alpha_{R}\rvert\rvert^{2}]"> |

## Dependencies

Dirac-GAN is written in [PyTorch 1.8.1](https://pytorch.org/). No GPU is required! All additional dependencies can be seen in the [`requirements.txt`](requirements.txt) file. To install all dependencies simply run:

```shellscript
pip install -r requirements.txt
```

[Older version of PyTorch](https://pytorch.org/get-started/previous-versions/) may also allows running the code without issues.

## Usage

The implementation provides a simple GUI to run all Dirac-GAN experiments with different settings. Simply run:

```shell script
python main.py
```

Set the desired parameters in the GUI and click on "Run training" to perform training. This could take a few seconds. If the training
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