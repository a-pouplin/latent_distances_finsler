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
    parser.add_argument("--num_geod", default=1, type=int)  # num of random geodesics to plot
    parser.add_argument("--iter_energy", default=300, type=int)  # num of steps to minimise energy func
    # data used
    parser.add_argument("--data", default="starfish", type=str)  # sphere or qPCR or font or starfish
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--model_folder", default="trained_models/", type=str)
    parser.add_argument("--exp_folder", default="plots/latent_indicatrices/", type=str)
    parser.add_argument("--exp_num", default="17", type=str)  # number of the experiment to use for plotting
    opts = parser.parse_args()
    return opts


def color_latent_data(X):
    fig = plt.figure(0)
    ax = plt.axes()
    # ax, im, _ = volume_heatmap(ax, gplvm_riemann, X, mode='vol_riemann', vmin=-1.5)
    ax.scatter(X[:, 0], X[:, 1], color="black", s=1)
    return ax


def far_away_points(X, num_points=50):
    index = np.argsort(np.linalg.norm(X, axis=1))[-num_points:]
    return X[index]


def get_angles(X):
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
    angles = np.arctan2(X[:, 1], X[:, 0])
    return np.rad2deg(angles)


def get_corner_starfish(X, thres=20):
    points = far_away_points(X)  # the num_pts far away points
    angles = get_angles(points)  # the corresponding angles in deg
    to_keep = np.empty((5, 2))
    for i in range(5):
        to_keep[i] = points[0]  # points array is updated each loop
        ind = (angles < angles[0] - thres) | (angles > angles[0] + thres)
        points = points[ind]
        angles = angles[ind]
    return to_keep


def plot_geodesics(X, Y, opts):
    # Energy function computed with riemannian metric
    optimizer = torch.optim.LBFGS
    eval_grid = 20
    dim_latent, dim_obs = X.shape[1], Y.shape[1]
    c_coords_riemann, c_obs_riemann = torch.empty((opts.num_geod, eval_grid, dim_latent)), np.empty(
        (opts.num_geod, eval_grid, dim_obs)
    )
    c_coords_finsler, c_obs_finsler = torch.empty((opts.num_geod, eval_grid, dim_latent)), np.empty(
        (opts.num_geod, eval_grid, dim_obs)
    )
    t = torch.linspace(0, 1, eval_grid)

    x1 = get_corner_starfish(X[:400])
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

    folderpath = os.path.abspath(str(opts.exp_folder) + "/" + str(opts.data) + "/" + str(opts.exp_num))
    create_folder(folderpath)
    print("--- figures saved in:", folderpath)

    # load previously training gplvm model
    model_folder = os.path.join(opts.model_folder, opts.data)
    model_saved = pickle_load(folder_path=model_folder, file_name="model_{}.pkl".format(opts.exp_num))
    model = model_saved["model"]
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
