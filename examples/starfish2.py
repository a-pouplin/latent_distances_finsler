# latent space visualisations for a trained GPLVM model

import argparse
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from stochman.curves import CubicSpline
from stochman.discretized_manifold import DiscretizedManifold
from stochman.geodesic import geodesic_minimizing_energy

from finsler.gplvm import Gplvm
from finsler.utils.data import make_sphere_surface, on_sphere
from finsler.utils.helper import create_filepath, create_folder, pickle_load
from finsler.visualisation.latent import volume_heatmap

matplotlib.rcParams["svg.fonttype"] = "none"


def get_args():
    parser = argparse.ArgumentParser()
    # manifold argments
    parser.add_argument("--num_geod", default=4, type=int)  # num of random geodesics to plot
    parser.add_argument("--iter_energy", default=400, type=int)  # num of steps to minimise energy func
    parser.add_argument("--save_manifold", default=False, type=bool)  # save model
    # data used
    parser.add_argument("--data", default="starfish2", type=str)  # sphere or starfish2 or vMF
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--exp_folder", default="plots/starfish2/", type=str)
    parser.add_argument("--model_folder", default="models/starfish2/", type=str)
    parser.add_argument("--model_title", default="model_10", type=str)
    opts = parser.parse_args()
    return opts


def color_latent_data(X):
    fig = plt.figure(0)
    ax = plt.axes()
    # ax, im, _ = volume_heatmap(ax, gplvm_riemann, X, mode='vol_riemann', vmin=-1.5)
    ax.scatter(X[:, 0], X[:, 1], color="black", s=1)
    return ax


def compute_heatmaps(n_grid):
    for mode in ["variance"]:
        print(mode)
        fig = plt.figure(0)
        ax = plt.axes()
        ax, im, hm_values, hm_ltt = volume_heatmap(ax, gplvm_riemann, X, mode, n_grid)  # , vmin=-3, vmax=0.5)
        ax.scatter(X[:, 0], X[:, 1], marker="o", edgecolors="black", s=1)
        fig.colorbar(im)
        plt.title("{}".format(mode))
        # plt.show()
        fig.savefig(
            os.path.join(opts.exp_folder, "heatmap_{}_{}{}.png".format(n_grid, mode, opts.title_model)), dpi=fig.dpi
        )
        fig.savefig(os.path.join(opts.final_plots, "heatmap_{}_{}_{}.svg".format(n_grid, mode, opts.data)))


def segment_lengths(batch_curves):
    """
    Computes the length of each segment of a batch of curves.
    :param batch_curves: (batch_size, num_points, dim) tensor
    :return lengths: (batch_size,) tensor
    """
    diffs = np.diff(batch_curves, axis=1)  # (batch_size, num_points-1, dim)
    norm_diff = np.linalg.norm(batch_curves, axis=2)  # (batch_size, num_points-1)
    lengths = np.sum(norm_diff, axis=1)  # (batch_size,)
    return lengths


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))

    folderpath = os.path.abspath(os.path.join(opts.exp_folder, opts.data))
    create_folder(folderpath)
    print("--- figures saved in:", folderpath)

    # load previously training gplvm model
    model = pickle_load(folder_path=f"{opts.model_folder}", file_name=f"{opts.model_title}.pkl")
    Y = model.y.data.numpy().transpose()
    X = model.X.data.numpy()
    lengthscale = model.kernel.lengthscale_unconstrained.data
    variance = model.kernel.variance_unconstrained.data
    print("Y shape: {} -- variance: {:.2f}, lengthscale: {}".format(Y.shape, variance, lengthscale))

    # get Riemannian and Finslerian metric
    model.X = model.X / torch.max(torch.abs(model.X))
    gplvm_riemann = Gplvm(model, mode="riemannian")
    gplvm_finsler = Gplvm(model, mode="finslerian")
    X = model.X.data

    # # plot observational space with sphere surface
    # y_random = gplvm_riemann.embed(torch.randn(1000, 2))[0].detach().numpy()
    # print(on_sphere(y_random, error=0.1))
    # # plot latent space
    # fig = plt.figure(0)
    # ax = plt.axes()
    # ax.scatter(X[:, 0], X[:, 1], marker="o", edgecolors="black", s=1)
    # plt.show()

    # # histogram of the norm of y
    # fig = plt.figure(0)
    # ax = plt.axes(projection="3d")
    # ax.set_box_aspect([1, 1, 1])
    # XS, YS, ZS = make_sphere_surface()  # for illustration
    # ax.plot_surface(XS, YS, ZS, shade=True, color="gray", alpha=0.1, zorder=0)
    # ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2], c="black", s=1, label="obs data", alpha=1)
    # ax.scatter3D(y_random[:,0], y_random[:,1], y_random[:,2], c='green', s=1, label='random latent data', alpha=1) # random points taken from the latent
    # plt.show()
    # raise

    # gte discretised manifold and save it
    modelpathriemann = os.path.join(opts.model_folder, "manifold_riemann_{}.pkl".format(opts.model_title))
    if opts.save_manifold:
        # with Discrete manifold
        with torch.no_grad():
            ran = torch.linspace(-1.0, 1.0, 50)  # the higher the number of points, the more accurate the geodesics
            gridX, gridY = torch.meshgrid([ran, ran], indexing="ij")
            grid = torch.stack((gridX.flatten(), gridY.flatten()), dim=1)  # 100x2
        manifold = DiscretizedManifold()
        manifold.fit(model=gplvm_riemann, grid=[ran, ran], batch_size=256)
        print("Manifold for Riemannian metric fitted. Saving model.....")
        # save manifold
        with open(modelpathriemann, "wb") as f:
            pickle.dump(manifold, f, pickle.HIGHEST_PROTOCOL)
        print("model saved !")

    modelpathfinsler = os.path.join(opts.model_folder, "manifold_finsler_{}.pkl".format(opts.model_title))
    if opts.save_manifold:
        # with Discrete manifold
        with torch.no_grad():
            ran = torch.linspace(-1.0, 1.0, 50)  # the higher the number of points, the more accurate the geodesics
            gridX, gridY = torch.meshgrid([ran, ran], indexing="ij")
            grid = torch.stack((gridX.flatten(), gridY.flatten()), dim=1)  # 100x2
        manifold = DiscretizedManifold()
        manifold.fit(model=gplvm_finsler, grid=[ran, ran], batch_size=256)
        print("Manifold for Finslerian metric fitted. Saving model.....")
        # save manifold
        with open(modelpathfinsler, "wb") as f:
            pickle.dump(manifold, f, pickle.HIGHEST_PROTOCOL)
        print("model saved !")

    with open(modelpathriemann, "rb") as file:
        manifoldR = pickle.load(file)
    assert isinstance(manifoldR, DiscretizedManifold), "Manifold should be of type DiscretizedManifold"
    with open(modelpathfinsler, "rb") as file:
        manifoldF = pickle.load(file)
    assert isinstance(manifoldF, DiscretizedManifold), "Manifold should be of type DiscretizedManifold"

    print("Manifold loaded !")
    p1 = torch.tensor([[0.8, 0.1], [0.2, 0.8], [-0.7, 0.4], [-0.5, -0.5], [0.3, -0.7]])
    p0 = torch.tensor([[0.3, -0.7], [0.8, 0.1], [0.2, 0.8], [-0.7, 0.4], [-0.5, -0.5]])
    opts.num_geod = p0.shape[0]

    splineR, _ = manifoldR.connecting_geodesic(p0, p1)
    splineF, _ = manifoldF.connecting_geodesic(p0, p1)
    splineE = CubicSpline(p0, p1)

    colors = sns.color_palette("viridis", n_colors=opts.num_geod)
    fig = plt.figure(1)
    ax = plt.axes()
    ax, im, hm_values, hm_ltt = volume_heatmap(ax, gplvm_riemann, X, mode="variance", n_grid=15)
    ax.scatter(X[:, 0], X[:, 1], marker="o", edgecolors="black", s=1, alpha=0.5)
    splineR.plot(color="purple", linewidth=1.5, zorder=1e5)
    splineF.plot(color="orange", linewidth=1.5, zorder=1e5)
    splineE.plot(color="gray", linewidth=1, linestyle="--", zorder=1e5)
    fig.colorbar(im)
    ax.set_aspect("equal")
    ax.axis("off")
    filename = "latent.svg"
    filepath = create_filepath(folderpath, filename)
    fig.savefig(filepath, dpi=fig.dpi)
    print("--- plot of the latent space saved as: {}".format(filepath))

    c_obs_riemann = gplvm_riemann.embed(splineR(torch.linspace(0, 1, 100)))[0].detach().numpy()
    c_obs_finsler = gplvm_finsler.embed(splineF(torch.linspace(0, 1, 100)))[0].detach().numpy()
    c_obs_euclid = gplvm_riemann.embed(splineE(torch.linspace(0, 1, 100)))[0].detach().numpy()

    print(c_obs_finsler.shape)
    print("Finsler", segment_lengths(c_obs_finsler))
    print("Riemann", segment_lengths(c_obs_riemann))
    print("Euclidean", segment_lengths(c_obs_euclid))

    fig = plt.figure(2)
    ax = plt.axes(projection="3d")
    ax.set_box_aspect([1, 1, 1])
    XS, YS, ZS = make_sphere_surface()  # for illustration
    ax.plot_surface(XS, YS, ZS, shade=True, color="gray", alpha=0.2, zorder=2)
    ax.scatter3D(
        Y[:, 0], Y[:, 1], Y[:, 2], label="observed data", marker="o", edgecolors="black", s=1, alpha=0.8, zorder=2
    )  # observed data points
    for i in range(opts.num_geod):
        ax.plot(
            c_obs_riemann[i, ::2, 0],
            c_obs_riemann[i, ::2, 1],
            c_obs_riemann[i, ::2, 2],
            c="purple",
            lw=1.5,
            # alpha=0.5,
            label="Riemann",
            zorder=2,
        )
        ax.plot(
            c_obs_finsler[i, 1::2, 0],
            c_obs_finsler[i, 1::2, 1],
            c_obs_finsler[i, ::2, 2],
            c="orange",
            lw=1.5,
            # ls=":",
            label="Finsler",
            zorder=2,
        )
        ax.plot(
            c_obs_euclid[i, 1::2, 0],
            c_obs_euclid[i, 1::2, 1],
            c_obs_euclid[i, ::2, 2],
            c="gray",
            lw=1.5,
            ls=":",
            label="Euclid",
            zorder=2,
        )
    ax.legend(bbox_to_anchor=(1.5, 1))
    ax.set_xlim((-1, 1)), ax.set_ylim((-1, 1)), ax.set_zlim((-1, 1))
    ax.grid(False)
    ax.axis("off")
    plt.show()

    filename = "observational.svg"
    filepath = create_filepath(folderpath, filename)
    fig.savefig(filepath, dpi=fig.dpi)
    print("--- plot of the observation space saved as: {}".format(filepath))
