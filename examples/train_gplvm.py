import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro.contrib.gp as gp
import pyro.ops.stats as stats
import torch

from finsler.gplvm import gplvm
from finsler.utils.data import (
    make_sphere_surface,
    make_torus_surface,
    on_sphere,
    on_torus,
)
from finsler.utils.helper import create_filepath, create_folder, pickle_save
from finsler.utils.pyro import initialise_kernel, iteration
from finsler.visualisation.latent import volume_heatmap


def get_args():
    parser = argparse.ArgumentParser()
    # initialisation
    parser.add_argument("--data", default="starfish_sphere", type=str)  # sphere or qPCR or font or starfish
    # optimisation arguments
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--iter", default=60000, type=int)
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--exp_folder", default="trained_models/", type=str)
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))

    folderpath = os.path.abspath(os.path.join(opts.exp_folder, opts.data))
    create_folder(folderpath)
    print("--- figures saved in:", folderpath)

    # initialise kernel
    Y, X, lengthscale, variance, noise = initialise_kernel(opts)
    print(X.shape)
    print(Y.shape)

    kernel = gp.kernels.Matern52(input_dim=2, variance=variance, lengthscale=lengthscale)
    # kernel = gp.kernels.RBF(input_dim=2, variance=variance, lengthscale=lengthscale)

    # Construct and train GP using pyro
    if Y.shape[0] < 64:
        Xu = stats.resample(X.clone(), Y.shape[0])
    else:
        Xu = stats.resample(X.clone(), 64)
    gpmodule = gp.models.SparseGPRegression(X, Y.t(), kernel, Xu, noise=noise, jitter=1e-4)
    model = gp.models.GPLVM(gpmodule)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr)
    log_params = iteration(model, optimizer, opts.iter, Y.shape[0])
    fig0, (ax0, ax1) = plt.subplots(2, 1)
    ax0.plot(np.log(log_params[0, :]))
    ax0.set_title("Loss: {:.2f}".format(np.mean(log_params[0, -10:])))
    ax1.plot(log_params[1, :], label="lengthscale")
    ax1.plot(log_params[2, :], label="variance")
    ax1.set_title("params: variance:{:.2f}, lengthscale:{:.2f}".format(log_params[1, -1], log_params[2, -1]))
    plt.show()
    print("\n")

    filename = "loss.png"
    filepath = create_filepath(folderpath, filename)
    fig0.savefig(filepath, dpi=fig0.dpi)
    print("--- plot of the loss saved as: {}".format(filepath))

    # save model
    model_save = {"model": model, "kernel": kernel, "Y": Y, "X": X}
    filename = "model.pkl"
    filepath = create_filepath(folderpath, filename)
    incr_filename = filepath.split("/")[-1]
    pickle_save(model_save, folderpath, file_name=incr_filename)
    print("--- model saved as: {}".format(filepath))

    # plot latent space with geodesics
    gplvm_riemann = gplvm(model, mode="riemannian")
    fig = plt.figure(1)
    ax = plt.axes()
    # ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax, im, _, _ = volume_heatmap(ax, gplvm_riemann, X, n_grid=16, mode="variance")

    if opts.data == "qPCR":
        URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
        df = pd.read_csv(URL, index_col=0)
        for lbl in df.index.unique():
            X_i = X[df.index == lbl]
            ax.scatter(X_i[:, 0], X_i[:, 1], label=lbl, s=2)
        ax.legend()
    else:
        ax.scatter(X[:, 0], X[:, 1], color="black", s=2)
    fig.colorbar(im)
    plt.title("Latent space")
    plt.show()

    filename = "latent.png"
    filepath = create_filepath(folderpath, filename)
    fig.savefig(filepath, dpi=fig.dpi)
    print("--- plot of the latent space saved as: {}".format(filepath))

    # plot observation space
    if opts.data in ["font_data", "qPCR"]:
        raise

    x_random = np.random.uniform(low=torch.min(X), high=torch.max(X), size=(100, 2))
    x_random = torch.tensor(x_random, dtype=torch.float32)
    y_random = gplvm_riemann.embed(x_random)[0].detach().numpy()  # y_random_mean

    if opts.data in ["starfish_sphere", "sphere", "sphere_holes", "vMF"]:
        print("Percentage of latent points on sphere: {:.2f}".format(100 * on_sphere(y_random)))

        fig = plt.figure(2)
        ax = plt.axes(projection="3d")
        XS, YS, ZS = make_sphere_surface()  # for illustration
        ax.plot_surface(XS, YS, ZS, shade=True, rstride=2, cstride=2, linewidth=0, color="gray", alpha=0.1)
        ax.plot_wireframe(XS, YS, ZS, rstride=2, cstride=2, linewidth=0.4, color="gray", alpha=0.2)
        ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2], s=1, label="observed data", zorder=0, alpha=0.2)  # observed data points
        ax.scatter3D(
            y_random[:, 0], y_random[:, 1], y_random[:, 2], c="green", s=1, label="random latent data", alpha=0.3
        )
        plt.show()

    elif opts.data == "torus":
        print("Percentage of latent points on torus: {:.2f}".format(100 * on_torus(y_random)))

        fig = plt.figure(2)
        ax = plt.axes(projection="3d")
        ax.set_box_aspect((np.ptp(Y[:, 0]), np.ptp(Y[:, 1]), np.ptp(Y[:, 2])))
        XS, YS, ZS = make_torus_surface()  # for illustration
        ax.plot_surface(XS, YS, ZS, shade=True, color="gray", alpha=0.3)
        # ax.scatter3D(Y[:,0], Y[:,1], Y[:,2], s=1, label='observed data', zorder=0, alpha=0.2) # observed data points
        ax.scatter3D(y_random[:, 0], y_random[:, 1], y_random[:, 2], c="black", s=1, label="random latent data")
        plt.show()

    filename = "observational.png"
    filepath = create_filepath(folderpath, filename)
    fig.savefig(filepath, dpi=fig.dpi)
    print("--- plot of the observational space saved as: {}".format(filepath))
