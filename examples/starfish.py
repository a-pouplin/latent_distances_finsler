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
from finsler.utils.data import make_sphere_surface, on_sphere
from finsler.utils.helper import create_filepath, create_folder, pickle_load
from finsler.visualisation.latent import volume_heatmap

matplotlib.rcParams["svg.fonttype"] = "none"


def get_args():
    parser = argparse.ArgumentParser()
    # manifold argments
    parser.add_argument("--num_geod", default=5, type=int)  # num of random geodesics to plot
    parser.add_argument("--iter_energy", default=100, type=int)  # num of steps to minimise energy func
    # data used
    parser.add_argument("--data", default="starfish", type=str)  # sphere or starfish or vMF
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--exp_folder", default="plots/starfish/", type=str)
    parser.add_argument("--model_folder", default="models/starfish/", type=str)
    parser.add_argument("--model_title", default="model_0", type=str)
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


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))

    folderpath = os.path.abspath(os.path.join(opts.exp_folder, opts.data))
    create_folder(folderpath)
    print("--- figures saved in:", folderpath)

    # load previously training gplvm model
    model = pickle_load(folder_path=f"{opts.model_folder}", file_name=f"{opts.model_title}.pkl")
    # model = model['model']
    Y = model.y.data.numpy().transpose()
    X = model.X.data.numpy()
    lengthscale = model.kernel.lengthscale_unconstrained.data
    variance = model.kernel.variance_unconstrained.data
    print("Y shape: {} -- variance: {:.2f}, lengthscale: {:.2f}".format(Y.shape, variance, lengthscale))

    # get Riemannian and Finslerian metric
    gplvm_riemann = Gplvm(model, mode="riemannian")
    gplvm_finsler = Gplvm(model, mode="finslerian")
    X = model.X.data.numpy()

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

    # computing geodesic curves along the starfish branches
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

    # plot latent space with geodesics
    colors = sns.color_palette("viridis", n_colors=opts.num_geod)
    fig = plt.figure(1)
    ax = plt.axes()
    ax, im, hm_values, hm_ltt = volume_heatmap(ax, gplvm_riemann, X, mode="vol_riemann", n_grid=5)
    for i in range(opts.num_geod):
        ax.plot(c_coords_riemann[i, ::2, 0], c_coords_riemann[i, ::2, 1], c=colors[i], lw=1, alpha=0.5)
        ax.plot(c_coords_finsler[i, 1::2, 0], c_coords_finsler[i, 1::2, 1], c=colors[i], lw=1, ls=":")
    ax.scatter(X[:, 0], X[:, 1], marker="o", edgecolors="black", s=1)
    fig.colorbar(im)
    plt.title("Geodesic in the latent space")
    plt.show()

    filename = "latent.png"
    filepath = create_filepath(folderpath, filename)
    fig.savefig(filepath, dpi=fig.dpi)
    print("--- plot of the latent space saved as: {}".format(filepath))

    # plot observational space with geodesics
    fig = plt.figure(2)
    ax = plt.axes(projection="3d")
    ax.set_box_aspect([1, 1, 1])
    XS, YS, ZS = make_sphere_surface()  # for illustration
    ax.plot_surface(XS, YS, ZS, shade=True, color="gray", alpha=0.5, zorder=0)
    ax.scatter3D(
        Y[:, 0], Y[:, 1], Y[:, 2], label="observed data", marker="o", edgecolors="black", s=1, zorder=2
    )  # observed data points
    # ax.scatter3D(y_random[:,0], y_random[:,1], y_random[:,2], c='green', s=1, label='random latent data', alpha=0.2) # random points taken from the latent
    for i in range(opts.num_geod):
        ax.plot(
            c_obs_riemann[i, ::2, 0],
            c_obs_riemann[i, ::2, 1],
            c_obs_riemann[i, ::2, 2],
            c=colors[i],
            lw=1,
            alpha=0.5,
            label="Riemann",
            zorder=9,
        )
        ax.plot(
            c_obs_finsler[i, 1::2, 0],
            c_obs_finsler[i, 1::2, 1],
            c_obs_finsler[i, ::2, 2],
            c=colors[i],
            lw=1,
            ls=":",
            label="Finsler",
            zorder=10,
        )
    ax.legend(bbox_to_anchor=(1.5, 1))
    ax.set_xlim((-1, 1)), ax.set_ylim((-1, 1)), ax.set_zlim((-1, 1))
    ax.grid(False)
    ax.axis("off")
    plt.show()

    filename = "observational.png"
    filepath = create_filepath(folderpath, filename)
    fig.savefig(filepath, dpi=fig.dpi)
    print("--- plot of the observation space saved as: {}".format(filepath))
