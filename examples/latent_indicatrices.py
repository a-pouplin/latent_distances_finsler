# latent space visualisations for a trained GPLVM model

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from stochman.curves import CubicSpline
from stochman.geodesic import geodesic_minimizing_energy

from finsler.gplvm import Gplvm
from finsler.utils.helper import create_filepath, create_folder, pickle_load
from finsler.visualisation.latent import (
    plot_indicatrices,
    plot_indicatrices_along_geodesic,
)

matplotlib.rcParams["svg.fonttype"] = "none"


def get_args():
    parser = argparse.ArgumentParser()
    # manifold argments
    parser.add_argument("--num_geod", default=0, type=int)  # num of random geodesics to plot
    parser.add_argument("--iter_energy", default=300, type=int)  # num of steps to minimise energy func
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--model_folder", default="models/starfish/", type=str)
    parser.add_argument("--exp_folder", default="plots/latent_indicatrices/", type=str)
    parser.add_argument("--model_title", default="model", type=str)  # number of the experiment to use for plotting
    opts = parser.parse_args()
    return opts


def color_latent_data(X):
    fig = plt.figure(0)
    ax = plt.axes()
    # ax, im, _ = volume_heatmap(ax, gplvm_riemann, X, mode='vol_riemann', vmin=-1.5)
    ax.scatter(X[:, 0], X[:, 1], color="black", s=1)
    return ax


def plot_geodesics(X, Y, opts):
    # Energy function computed with riemannian metric
    optimizer = torch.optim.LBFGS
    eval_grid = 16
    dim_latent, dim_obs = X.shape[1], Y.shape[1]
    c_coords_riemann, c_obs_riemann = torch.empty((opts.num_geod, eval_grid, dim_latent)), np.empty(
        (opts.num_geod, eval_grid, dim_obs)
    )
    c_coords_finsler, c_obs_finsler = torch.empty((opts.num_geod, eval_grid, dim_latent)), np.empty(
        (opts.num_geod, eval_grid, dim_obs)
    )
    t = torch.linspace(0, 1, eval_grid)

    x1 = [[-2, -2], [1.0, -2.5], [-2, 1.5], [1, 2.5], [2.5, 0]]
    x0 = np.tile([0, 0], (len(x1), 1))
    for i in range(opts.num_geod):
        start, ends = torch.tensor([x0[i]], dtype=torch.float), torch.tensor([x1[i]], dtype=torch.float)

        print("\n")
        print("Riemannian")
        curve_r = CubicSpline(start, ends, requires_grad=True)
        geodesic_minimizing_energy(curve_r, gplvm_riemann, optimizer, opts.iter_energy, eval_grid)

        print("\n")
        print("Finslerian")
        curve_f = CubicSpline(start, ends, requires_grad=True)
        geodesic_minimizing_energy(curve_f, gplvm_finsler, optimizer, opts.iter_energy, eval_grid)

        with torch.no_grad():
            c_coords_riemann[i, :, :] = curve_r(t)
            c_coords_finsler[i, :, :] = curve_f(t)
            c_obs_riemann[i, :, :], _ = gplvm_riemann.embed(c_coords_riemann[i, :, :].squeeze())
            c_obs_finsler[i, :, :], _ = gplvm_finsler.embed(c_coords_finsler[i, :, :].squeeze())

    return c_coords_riemann, c_coords_finsler, c_obs_riemann, c_obs_finsler


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))

    folderpath = os.path.abspath(str(opts.exp_folder) + "/" + str(opts.model_title))
    create_folder(folderpath)
    print("--- figures saved in:", folderpath)

    # load previously training gplvm model
    model = pickle_load(folder_path=f"{opts.model_folder}", file_name=f"{opts.model_title}.pkl")
    Y = model.y.data.numpy().transpose()
    X = model.X.data.numpy()
    lengthscale = model.kernel.lengthscale_unconstrained.data
    variance = model.kernel.variance_unconstrained.data
    print("Y shape: {} -- variance: {:.2f}, lengthscale: {:.2f}".format(Y.shape, variance, lengthscale))

    # get Riemannian and Finslerian metric
    gplvm_riemann = Gplvm(model, mode="riemannian")
    gplvm_finsler = Gplvm(model, mode="finslerian")

    # plot latent space with indicatrices
    fig = plt.figure(1)
    ax = plt.axes()

    # plot along the geodesic
    if opts.num_geod != 0:
        ccr, ccf, _, _ = plot_geodesics(X, Y, opts)
        colors = sns.color_palette("viridis", n_colors=opts.num_geod)
        for i in range(opts.num_geod):
            ax.plot(ccr[i, :, 0], ccr[i, :, 1], c=colors[i], lw=1, alpha=0.5)
            ax.plot(ccf[i, :, 0], ccf[i, :, 1], c=colors[i], lw=1, ls=":")
            ax = plot_indicatrices_along_geodesic(ax, gplvm_riemann, ccr[i, :], ccf[i, :], n_grid=8)
    # or in the entire latent space
    else:
        ax = plot_indicatrices(ax, gplvm_riemann, X, n_grid=8)

    ax.scatter(X[:, 0], X[:, 1], marker="o", edgecolors="black", s=1)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # fig.colorbar(im)
    plt.title("Latent space with indicatrices")
    plt.show()

    filename = "latent.png"
    filepath = create_filepath(folderpath, filename)
    fig.savefig(filepath, dpi=fig.dpi)
    print("--- indiactrices in latent space saved as: {}".format(filepath))
