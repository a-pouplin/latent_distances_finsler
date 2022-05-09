from time import time

import numpy as np
import pandas as pd
import pyro
import spio
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from finsler.utils.data import (
    make_sphere_points,
    remove_points,
    sample_sphere,
    sample_vMF,
)


def update(optimizer, model, n_samples):
    loss_fn = pyro.infer.TraceMeanField_ELBO().differentiable_loss  # Trace_ELBO
    loss = loss_fn(model.model, model.guide) / n_samples
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def iteration(model, optimizer, num_iter, n_samples):
    losses = []
    for iter in range(num_iter):
        init_time = time()
        loss = update(optimizer, model, n_samples)
        losses.append(loss.item())
        _time = time() - init_time
        if iter % 100 == 0:
            print("Batch: {}/{} - loss {:.2f} -  time {:.3f}".format(iter, num_iter, loss, _time), end="\r")
    return losses


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


def initialise_kernel(data):
    if data == "sphere":
        Y = make_sphere_points(400)
        print(Y.shape)
        Y = torch.tensor(Y, dtype=torch.float32)
        X = get_prior(Y=Y, init_name="iso")
        lengthscale = torch.tensor(1.0, requires_grad=True)
        variance = torch.tensor(0.1, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=True)

    elif data == "qPCR":
        URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
        df = pd.read_csv(URL, index_col=0)
        Y = torch.tensor(df.values, dtype=torch.get_default_dtype())
        X = get_prior(Y=Y.t(), init_name="qPCR", df=df)
        # lengthscale = torch.ones(2, requires_grad=True)
        # variance = None
        # noise = torch.tensor(0.001, requires_grad=False)
        lengthscale = torch.tensor(1.0, requires_grad=True)
        variance = torch.tensor(0.1, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=True)

    elif data == "sphere_holes":
        points = sample_sphere(1000)
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
        variance = torch.tensor(0.5, requires_grad=True)
        noise = torch.tensor(0.01, requires_grad=False)

    elif data == "vMF":
        num_blobs, num_samples = 10, 100
        Y = np.zeros((num_blobs * num_samples, 3))
        mus = np.random.uniform(low=-1, high=1, size=(num_blobs, 3))
        for i, mu in enumerate(mus):
            Y[num_samples * i : num_samples * (i + 1)] = sample_vMF(mu=mu, kappa=50.0, num_samples=num_samples)
        Y = torch.tensor(Y, dtype=torch.float32)
        X = get_prior(Y=Y, init_name="iso")
        lengthscale = torch.tensor(0.5, requires_grad=True)
        variance = torch.tensor(0.1, requires_grad=True)
        noise = torch.tensor(0.001, requires_grad=True)

    # elif data == 'font':
    #     letter = "G"
    #     mat = spio.loadmat('../data/fonts_matlab/{}.mat'.format(letter), squeeze_me=True)
    #     Y, X, lengthscale, variance, noise = font_prior(mat) # still need to get prior ?
    return Y, X, lengthscale, variance, noise
