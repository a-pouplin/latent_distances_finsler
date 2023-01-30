# Identifying latent distances with Finslerian geometry

You will need the [stochman](https://github.com/MachineLearningLifeScience/stochman) and [pyro](https://pyro.ai/) packages to run this code.

In order to obtain the latent space for starfish data:

* train the GPLVM: `examples/train_gplvm.py`. The model should be saved in `.pkl` in the `trained_models` folder.
* plot geodesics: `examples/sphere.py`. Two files: `latent.png` and `observational.png` should be created in the `plots/spheres` folder.
