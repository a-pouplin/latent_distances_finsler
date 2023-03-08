# Identifying latent distances with Finslerian geometry

### Requirements
You will need the [stochman](https://github.com/MachineLearningLifeScience/stochman) and [pyro](https://pyro.ai/) packages to run this code.

You can install all the requirements in two ways: 
* `make requirements`, or:
* `pip install -r requirements.txt` and `pip install -e .`

### Training the model
In order to train the GPLVM on the starfish data, we use [wandb](https://wandb.ai/site). You may either want to modify the code or to login to your wandb account.
* To train the model: `make train`

### Experiments
The GPLVM models for the starfish, qPCR and font data have been saved. The figures can be obtained with: 
* `make figure3`. The indicatrices will be plotted in the background and along one geodesic.
* `make figure4`. The heatmaps will be computed, this code is time consuming. You might want to lower `n_grid` from 32 to 8.
* `make figure5`. Illustration of theoretical results are plotted.
* `make figure7`. The latent space of the fontdata and the qPCR data are plotted.

Note that the `starfish/model.pkl` is not exactly similar to the one from the paper, but it doesn't change the conclusion and the main findings. 
