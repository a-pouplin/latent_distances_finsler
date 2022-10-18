from time import time

import numpy as np
import pandas as pd
import pyro
import scipy
import scipy.io as sio
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from finsler.utils.data import (
    highdim_circles,
    highdim_starfish,
    make_sphere_points,
    make_torus_points,
    remove_points,
    sample_sphere,
    sample_vMF,
    starfish_2sphere,
)


def update(optimizer, model, n_samples):
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss  # Trace_ELBO
    loss = loss_fn(model.model, model.guide) / n_samples
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def iteration(model, optimizer, num_iter, n_samples):
    log_params = np.empty((3, num_iter))
    for iter in range(num_iter):
        init_time = time()
        loss = update(optimizer, model, n_samples)
        _time = time() - init_time

        log_params[0, iter] = loss.item()
        lengthscale = model.base_model.kernel.lengthscale_unconstrained
        variance = model.base_model.kernel.variance_unconstrained
        log_params[1, iter] = lengthscale.detach().numpy()
        log_params[2, iter] = variance.detach().numpy()

        # with torch.no_grad():
        #     if iter > int(0.3*num_iter):
        #         torch.clamp_(lengthscale, min=0, max=0.5)
        #         torch.clamp_(variance, min=0.3)

        max_grad = max([p.grad.abs().max() for p in model.parameters()])
        if iter % 100 == 0:
            print(
                "Batch: {}/{} - loss {:.2f} - grad {:.2f} - lengthscale {:.2f} -  variance {:.3f}".format(
                    iter, num_iter, loss, max_grad, lengthscale, variance
                ),
                end="\r",
            )
        # if iter > 5000:
        #     with torch.no_grad():
        #         torch.clamp_(lengthscale, min=0, max=1.)
        #         torch.clamp_(variance, min=0.1)
        # if iter > 10000:
        #     with torch.no_grad():
        #         torch.clamp_(lengthscale, min=0, max=0.5)
        if (loss < 0.01) and (max_grad < 0.01):
            break
    return log_params


def get_prior(Y, init_name, df=None, latent_dim=2):
    if "pca" in init_name:
        X_init = PCA(n_components=latent_dim).fit_transform(Y)
    elif "iso" in init_name:
        X_init = Isomap(n_components=latent_dim).fit_transform(Y)
        scale = np.mean(np.abs(np.array([np.max(X_init), np.min(X_init)])))
        X_init = X_init / scale  # not sure about this part
    elif "qPCR" in init_name:
        capture_time = Y.new_tensor([int(cell_name.split(" ")[0]) for cell_name in df.index.values])
        time = capture_time.log2() / 6
        X_init = torch.zeros(Y.size(1), 2)  # shape: 437 x 2
        X_init[:, 0] = time
    return torch.tensor(X_init, dtype=torch.float32)


def initialise_kernel(opts):
    data = opts.data
    if data == "sphere":
        Y = make_sphere_points(500)
        Y = torch.tensor(Y, dtype=torch.float32)
        X = get_prior(Y=Y, init_name="iso")
        lengthscale = torch.tensor(1.0, requires_grad=True)
        variance = torch.tensor(5.0, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=False)

    elif data == "qPCR":
        URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
        df = pd.read_csv(URL, index_col=0)
        Y = torch.tensor(df.values, dtype=torch.get_default_dtype())
        X = get_prior(Y=Y.t(), init_name="qPCR", df=df)
        # lengthscale = torch.ones(2, requires_grad=True)
        # variance = None
        # noise = torch.tensor(0.001, requires_grad=False)
        lengthscale = torch.tensor(1.0, requires_grad=True)
        variance = torch.tensor(10.0, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=False)

    elif data == "sphere_holes":
        points = sample_sphere(500)
        Y = points
        Y = remove_points(Y, center=[-1, 0, 0], radius=0.5)
        Y = remove_points(Y, center=[0, -1, 0], radius=0.5)
        Y = remove_points(Y, center=[0, 0, -1], radius=0.5)
        Y = remove_points(Y, center=[1, 0, 0], radius=0.5)
        Y = remove_points(Y, center=[0, 1, 0], radius=0.5)
        Y = remove_points(Y, center=[0, 0, 1], radius=0.5)
        Y = torch.tensor(Y, dtype=torch.float32)
        X = get_prior(Y=Y, init_name="iso")
        lengthscale = torch.tensor(0.1, requires_grad=True)
        variance = torch.tensor(10.0, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=False)

    elif data == "vMF":
        num_blobs, num_samples = 5, 100
        Y = np.zeros((num_blobs * num_samples, 3))
        mus = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, -1], [0, -1, 0]]
        # mus = np.random.uniform(low=-1, high=1, size=(num_blobs, 3))
        for i, mu in enumerate(mus):
            Y[num_samples * i : num_samples * (i + 1)] = sample_vMF(mu=mu, kappa=40.0, num_samples=num_samples)
        Y = torch.tensor(Y, dtype=torch.float32)
        X = get_prior(Y=Y, init_name="iso")
        lengthscale = torch.tensor(0.1, requires_grad=True)
        variance = torch.tensor(10.0, requires_grad=True)
        noise = torch.tensor(0.01, requires_grad=False)

    elif data == "font":
        file = "Output{}.mat".format(opts.model_title)
        print(file)
        mat = sio.loadmat("/Users/alpu/Documents/code/data/fonts_matlab/{}".format(file), squeeze_me=True)
        Y = mat["Y"]
        num_letters, _ = Y.shape
        if opts.model_title == "_1_g":
            Y = Y.reshape(num_letters, 6, -1)
        else:
            Y = Y.reshape(num_letters, 4, -1)

        samp = int(Y.shape[2] / 32)
        Y = Y[:, :, ::samp]
        Y = Y.reshape(num_letters, -1)
        Y = torch.tensor(Y, dtype=torch.float32)
        X = get_prior(Y=Y, init_name="pca")
        lengthscale = torch.tensor(1.0, requires_grad=True)
        variance = torch.tensor(10.0, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=False)

    # elif data == "starfish": # dim = 3
    #     Y, X = make_starfish()
    #     Y = torch.tensor(Y, dtype=torch.float32)
    #     X = get_prior(Y=Y, init_name="pca")
    #     lengthscale = torch.tensor(0.75, requires_grad=True)
    #     variance = torch.tensor(20., requires_grad=True)
    #     noise = torch.tensor(0.01, requires_grad=True)

    elif data == "starfish":  # high dimensions
        Y, X = highdim_starfish(dim=3, num_classes=5, num_per_class=100)
        Y = torch.tensor(Y, dtype=torch.float32)
        blur = torch.normal(0, 1, size=X.shape)
        X = torch.tensor(X, dtype=torch.float32) + 0.1 * blur
        # X = get_prior(Y=Y, init_name="pca")
        lengthscale = torch.tensor(0.1, requires_grad=True)
        variance = torch.tensor(10.0, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=False)

    elif data == "starfish_sphere":  # high dimensions
        Y, X = starfish_2sphere(num_classes=5, num_per_class=200)
        Y = torch.tensor(Y, dtype=torch.float32)
        blur = torch.normal(0, 1, size=X.shape)
        # X = torch.tensor(X, dtype=torch.float32) + 0.1*blur
        X = get_prior(Y=Y, init_name="pca")
        lengthscale = torch.tensor(10.0, requires_grad=True)
        variance = torch.tensor(10.0, requires_grad=True)
        noise = torch.tensor(0.0001, requires_grad=False)

    elif data == "concentric_circles":  # high dimensions
        Y, X = highdim_circles(dim=5, num_per_circle=250, num_circles=2)
        Y = torch.tensor(Y, dtype=torch.float32)
        blur = torch.normal(0, 1, size=X.shape)
        X = torch.tensor(X, dtype=torch.float32) + 0.01 * blur
        # X = get_prior(Y=Y, init_name="iso")
        lengthscale = torch.tensor(0.1, requires_grad=True)
        variance = torch.tensor(10.0, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=False)

    elif data == "proteins":
        from data.data_proteins import data

        data, df, _, _ = data()
        data = data[:, :, 0]
        Y = torch.tensor(data, dtype=torch.float32)
        X = get_prior(Y=Y, init_name="iso")
        lengthscale = torch.tensor(0.1, requires_grad=True)
        variance = torch.tensor(10.0, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=False)

    elif data == "torus":
        Y = make_torus_points(1000)
        Y = torch.tensor(Y, dtype=torch.float32)
        X = get_prior(Y=Y, init_name="iso")
        lengthscale = torch.tensor(10.0, requires_grad=True)
        variance = torch.tensor(10.0, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=False)

    elif data == "omniglot":
        from data.torchdata import Omniglot

        data = Omniglot()
        Y, label = data.x, data.y
        Y = torch.tensor(Y, dtype=torch.float32)
        X = get_prior(Y=Y, init_name="iso")
        lengthscale = torch.tensor(1.0, requires_grad=True)
        variance = torch.tensor(10.0, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=False)

    return Y, X, lengthscale, variance, noise
