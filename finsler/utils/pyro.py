from time import time

import numpy as np
import pandas as pd
import pyro
import scipy.io as sio
import torch
from pyro.infer import SVI, TraceMeanField_ELBO
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from finsler.utils.data import (
    cheesboard_2sphere,
    cheese_2sphere,
    concentric_2sphere,
    highdim_starfish_alacilie,
    starfish_2sphere,
)


def log_likelihood(model, data, latent_dim=2):
    raise NotImplementedError


def update(optimizer, model, n_samples, mode="tracemeanfield"):
    print(mode)
    if mode == "pyro":
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss  # Trace_ELBO
        loss = loss_fn(model.model, model.guide) / n_samples
    elif mode == "tracemeanfield":
        svi = SVI(model.model, model.guide, optimizer, loss=TraceMeanField_ELBO())
        loss = svi.step() / n_samples
    elif mode == "likelihood":
        loss = -log_likelihood(model, n_samples)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def iteration(model, optimizer, num_iter, n_samples):
    log_params = np.empty((3, num_iter))
    num_data, latent_dim = model.X_loc.shape

    for iter in range(num_iter):
        init_time = time()
        loss = update(optimizer, model, n_samples)
        _time = time() - init_time

        log_params[0, iter] = loss.item()
        lengthscale = model.base_model.kernel.lengthscale_unconstrained.data
        variance = model.base_model.kernel.variance_unconstrained.data
        log_params[1, iter] = lengthscale.detach().numpy()
        log_params[2, iter] = variance.detach().numpy()

        with torch.no_grad():
            torch.clamp_(lengthscale, min=0.1)
            torch.clamp_(variance, min=0.1)

        max_grad = max([p.grad.abs().max() for p in model.parameters()])
        if iter % 100 == 0:
            print(
                "Batch: {}/{} - loss {:.2f} - grad {:.2f} - lengthscale {:.2f} -  variance {:.3f}".format(
                    iter, num_iter, loss, max_grad, lengthscale, variance
                ),
                end="\r",
            )
        if (loss < 0.01) and (max_grad < 0.01):
            break
    return log_params


def get_prior(Y, init_name, df=None, latent_dim=2):
    if "pca" in init_name:
        X_init = PCA(n_components=latent_dim).fit_transform(Y)
    elif "iso" in init_name:
        X_init = Isomap(n_components=latent_dim).fit_transform(Y)
        scale = np.mean(np.abs(np.array([np.max(X_init), np.min(X_init)])))
        X_init = X_init / scale
    elif "qPCR" in init_name:
        capture_time = Y.new_tensor([int(cell_name.split(" ")[0]) for cell_name in df.index.values])
        time = capture_time.log2() / 6
        X_init = torch.zeros(Y.size(1), 2)  # shape: 437 x 2
        X_init[:, 0] = time
    return torch.tensor(X_init, dtype=torch.float32)


def initialise_kernel(data):
    if data == "qPCR":
        URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
        df = pd.read_csv(URL, index_col=0)
        Y = torch.tensor(df.values, dtype=torch.get_default_dtype())
        X = get_prior(Y=Y.t(), init_name="qPCR", df=df)
        return X, Y

    elif data == "font":
        mat = sio.loadmat("data/font_f.mat", squeeze_me=True)
        Y = mat["Y"]
        num_letters, _ = Y.shape
        Y = Y.reshape(num_letters, 4, -1)
        samp = int(Y.shape[2] / 32)
        Y = Y[:, :, ::samp]
        Y = Y.reshape(num_letters, -1)
        Y = torch.tensor(Y, dtype=torch.float32)
        X = get_prior(Y=Y, init_name="pca")
        return X, Y

    elif data == "starfish":  # high dimensions
        Y, X_true = starfish_2sphere(num_classes=5, num_per_class=100)
        Y = torch.tensor(Y, dtype=torch.float32)
        blur = torch.normal(0, 1, size=X_true.shape)
        X = torch.tensor(X_true, dtype=torch.float32)  # + 0.01 * blur
        # X = get_prior(Y=Y, init_name="pca")
        return Y, X, X_true

    elif data == "cheese":  # high dimensions
        Y, X_true = cheese_2sphere(num_holes=1, radius_holes=0.5, num_data=2000, center_holes=np.zeros(2))
        Y = torch.tensor(Y, dtype=torch.float32)
        blur = torch.normal(0, 1, size=X_true.shape)
        X = torch.tensor(X_true, dtype=torch.float32)  # + 0.01 * blur
        # X = get_prior(Y=Y, init_name="pca")
        return Y, X, X_true

    elif data == "chessboard":  # high dimensions
        Y, X_true = cheesboard_2sphere(num_grids=5, num_points=2000)
        Y = torch.tensor(Y, dtype=torch.float32)
        blur = torch.normal(0, 1, size=X_true.shape)
        X = torch.tensor(X_true, dtype=torch.float32)  # + 0.01 * blur
        # X = get_prior(Y=Y, init_name="pca")
        return Y, X, X_true

    elif data == "starfish2":  # high dimensions
        Y, X_true = starfish_2sphere(num_classes=5, num_per_class=200)
        Y = torch.tensor(Y, dtype=torch.float32)
        blur = torch.normal(0, 1, size=X_true.shape)
        # X = torch.tensor(X_true, dtype=torch.float32) #+ 0.01 * blur
        X = get_prior(Y=Y, init_name="pca")
        return Y, X, X_true

    elif data == "starfish_cilie":  # high dimensions
        Y, X_true = highdim_starfish_alacilie(num_classes=5, num_per_class=100)
        Y = torch.tensor(Y, dtype=torch.float32)
        blur = torch.normal(0, 1, size=X_true.shape)
        # X =  X_true.clone().detach().requires_grad_(True) #+ 0.01 * blur
        X = get_prior(Y=Y, init_name="pca")
        return Y, X, X_true

    elif data == "concentric_circles":  # high dimensions
        Y, X_true = concentric_2sphere(num_classes=4, num_per_class=100)
        Y = torch.tensor(Y, dtype=torch.float32)
        blur = torch.normal(0, 1, size=X_true.shape)
        # X =  X_true.clone().detach().requires_grad_(True) #+ 0.01 * blur
        X = get_prior(Y=Y, init_name="pca")
        return Y, X, X_true
