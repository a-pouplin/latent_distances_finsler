# Identifying latent distances with Finslerian geometry

Code for the paper **[Identifying latent distances with Finslerian geometry](https://openreview.net/forum?id=Q2Gi0TUAdS)** (Alison Pouplin, David Eklund, Carl Henrik Ek, Søren Hauberg), published in *Transactions on Machine Learning Research*, 2023. A preprint is available on [arXiv](https://arxiv.org/abs/2212.10010).

### Summary
Generative models such as the GPLVM map a low-dimensional latent space to the data space through a *stochastic* function. The Riemannian metric pulled back through this map is therefore stochastic too, and the usual fix is to replace it by its expectation (the *expected Riemannian metric*). The geodesics of this expected metric, however, do not minimise the expected length of curves.

This repository implements the alternative studied in the paper:
* The **Finsler metric** obtained by taking the expectation of the stochastic norm directly, whose geodesics minimise the expected curve length. For Gaussian processes it has a closed form in terms of the non-central Nakagami distribution.
* Tools to compare both metrics in the latent space of a trained GPLVM: indicatrices, geodesics, and Busemann-Hausdorff volume measures.
* Experiments reproducing the paper's figures on synthetic data (pinwheel and concentric circles on a sphere), the font and qPCR datasets, and MNIST / FashionMNIST.

The main result is that both metrics converge to each other at a rate of O(1/D), with D the dimension of the data space, which justifies using the expected Riemannian metric in practice.

### Requirements
You will need the [stochman](https://github.com/MachineLearningLifeScience/stochman) and [pyro](https://pyro.ai/) packages to run this code.

You can install all the requirements in two ways:
* `make requirements`, or:
* `pip install -r requirements.txt` and `pip install -e .`

### Project structure
* `finsler/`: the library. GPLVM and stochastic active sets (`gplvm.py`, `sasgp.py`), kernels and likelihoods, the Finsler and Riemannian distributions (`distributions.py`), and plotting utilities.
* `examples/`: scripts that reproduce the figures of the paper (see below).
* `models/`: training script and saved GPLVM models for each dataset.
* `data/`: the datasets used in the experiments.
* `tests/`: unit tests.

### Training the model
In order to train the GPLVM on the starfish data, we use [wandb](https://wandb.ai/site). You may either want to modify the code or to login to your wandb account.
* To train the model: `make train`

### Experiments
The GPLVM models for the starfish, qPCR and font data have been saved. The figures can be obtained with:
* `make figure2`. The indicatrices will be plotted in the background and along one geodesic.
* `make figure3`. The heatmaps will be computed, this code is time consuming.
* `make figure4`. Illustration of theoretical results are plotted.
* `make figure5`. The latent spaces of the synthetic data are plotted.
* `make figure6`. The latent spaces of the qPCR data is plotted.
* `make figure7`. The latent spaces of the MNIST and fashion MNIST are plotted.

Note that the figure for the fontdata might not be exactly similar to the one from the paper, but it doesn't change the conclusion and the main findings.

### Citation
If you use this code, please cite:
```bibtex
@article{pouplin2023identifying,
  title={Identifying latent distances with Finslerian geometry},
  author={Pouplin, Alison and Eklund, David and Ek, Carl Henrik and Hauberg, S{\o}ren},
  journal={Transactions on Machine Learning Research},
  year={2023},
  url={https://openreview.net/forum?id=Q2Gi0TUAdS}
}
```
