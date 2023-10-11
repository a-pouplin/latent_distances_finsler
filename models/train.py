import argparse
import os
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.ops.stats as stats
import torch

import wandb
from finsler.gplvm import Gplvm
from finsler.utils.data import on_sphere
from finsler.utils.helper import (
    create_filepath,
    create_folder,
    pickle_load,
    pickle_save,
)
from finsler.utils.pyro import initialise_kernel


def get_args():
    parser = argparse.ArgumentParser()
    # initialisation
    parser.add_argument("--data", default="concentric_circles", type=str)  # train for concentric_circles
    parser.add_argument("--sweep", default=False, type=bool)
    parser.add_argument("--exp_folder", default="models/", type=str)
    parser.add_argument("--train_pyro", default=False, type=bool)
    opts = parser.parse_args()
    return opts


def make(config, data):
    # initialise kernel
    Y, X_prior, X_true = initialise_kernel(data)
    lengthscale = config.lengthscale * torch.ones(2, dtype=torch.float32)
    variance = torch.tensor(config.variance, dtype=torch.float32)
    noise = torch.tensor(config.noise, dtype=torch.float32)

    kernel = getattr(gp.kernels, config.kernel)(input_dim=2, variance=variance, lengthscale=lengthscale)
    # Construct and train GP using pyro
    # X = Parameter(X_prior.clone())
    X = X_prior
    # adding inducing points
    if Y.shape[0] < 64:
        Xu = stats.resample(X_prior.clone(), Y.shape[0])
    else:
        Xu = stats.resample(X_prior.clone(), 64)
    gpmodule = gp.models.SparseGPRegression(X, Y.t(), kernel, Xu, noise=noise, jitter=1e-4)
    # gpmodule.X = pyro.nn.PyroSample(dist.Normal(X_prior, 0.1).to_event()) # no batch shape
    # gpmodule.autoguide("X", dist.Normal)
    model = gp.models.GPLVM(gpmodule)
    return model, Y, X_true


def test(model, X_true):
    # parameters
    X = model.X.data.numpy()

    # initialise gplvm
    gplvm_riemann = Gplvm(model, mode="riemannian")

    x_random = np.random.uniform(low=np.min(X), high=np.max(X), size=(1000, 2))
    x_random = torch.tensor(x_random, dtype=torch.float32)
    y_random = gplvm_riemann.embed(x_random)[0].detach().numpy()  # y_random_mean
    acc_obs = on_sphere(y_random)  # 1 if on sphere, 0 if not

    # compute difference between true and trained latent points
    # not necessary a meaningful metric
    dist = np.mean(np.linalg.norm(X_true - X, axis=-1))

    return acc_obs, dist


def save_model(model, folderpath):
    model = model.base_model
    filepath = create_filepath(folderpath, "model.pkl")
    incr_filename = filepath.split("/")[-1]
    pickle_save(model, folderpath, file_name=incr_filename)
    print("--- model saved as: {}".format(filepath))
    return incr_filename


def update(optimizer, model, n_samples, X_true=None):
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss  # Trace_ELBO
    loss = loss_fn(model.model, model.guide) / n_samples
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def model_pipeline(config=None):
    # initialise wandb
    print(config)
    wandb.init(config=config)
    config = wandb.config

    # initialise model
    model, Y, X_true = make(config, data=opts.data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if opts.train_pyro:
        print("training pyro model ........")
        losses = gp.util.train(model, num_steps=config.iter, optimizer=optimizer)
        print(".......... training done !")
        plt.plot(losses)
        plt.show()
    else:
        # train model
        for iter in range(config.iter):
            loss = update(optimizer, model, Y.shape[0])

            lengthscale = model.base_model.kernel.lengthscale_unconstrained.data
            variance = model.base_model.kernel.variance_unconstrained.data
            with torch.no_grad():
                torch.clamp_(lengthscale, min=0.01, max=0.5)
                torch.clamp_(variance, min=0.1)

            # max_grad = max([p.grad.abs().max() for p in model.parameters()])
            if iter % 100 == 0:
                print(
                    "Batch: {}/{} - loss {:.2f}  - lengthscale [{:.2f},{:.2f}] -  variance {:.3f}".format(
                        iter, config.iter, loss, lengthscale[0], lengthscale[1], variance
                    ),
                    end="\r",
                )
            wandb.log({"loss": loss, "lengthscale": lengthscale, "variance": variance})

    # save model
    incr_filename = save_model(model, folderpath)
    print("--- model saved in:", folderpath, incr_filename)

    # load model
    model = pickle_load(folderpath, incr_filename)

    # test model
    acc_obs, dist = test(model, X_true)

    wandb.log({"pts_on_sphere": acc_obs, "diff_latent": dist})
    table = wandb.Table(data=model.X.data.numpy(), columns=["xp", "yp"])
    wandb.log({"latent space": wandb.plot.scatter(table, "xp", "yp", title="Latent space")})

    print("number of points on sphere:", acc_obs)
    print("distance between true and trained latent points:", dist)
    print("-" * 50)
    return model


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))

    folderpath = os.path.abspath(os.path.join(opts.exp_folder, opts.data))
    create_folder(folderpath)

    if opts.sweep:
        print("--- sweep mode ---")
        # sweep config parameters
        sweep_dict = {
            "lr": {"values": [1e-2]},
            "iter": {"distribution": "int_uniform", "min": 10000, "max": 15000},
            "kernel": {"values": ["Matern32"]},
            "lengthscale": {"distribution": "uniform", "min": 0.001, "max": 0.05},
            "variance": {"distribution": "uniform", "min": 1.0, "max": 2.0},
            "noise": {"values": [1e-4]},
        }

        sweep_config = {
            "method": "random",
            "name": "concentric_circles",
            "metric": {"name": "dist", "goal": "minimize"},
            "parameters": sweep_dict,
        }
        pprint.pprint(sweep_config)
        sweep_id = wandb.sweep(sweep=sweep_config, project="latent_finsler_distances")
        wandb.agent(sweep_id, function=model_pipeline, count=50)
    else:
        print("--- single run mode ---")
        # best params
        config_params = {
            "lr": 1e-3,
            "iter": 17000,
            "kernel": "Matern32",
            "lengthscale": 0.24,
            "variance": 0.95,
            "noise": 1e-4,
        }
        model_pipeline(config=config_params)
