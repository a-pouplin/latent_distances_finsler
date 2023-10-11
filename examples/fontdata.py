# latent space visualisations for a trained GPLVM model

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from stochman.curves import CubicSpline
from stochman.geodesic import geodesic_minimizing_energy

from finsler.gplvm import Gplvm
from finsler.utils.helper import create_filepath, create_folder, pickle_load
from finsler.visualisation.latent import volume_heatmap


def get_args():
    parser = argparse.ArgumentParser()
    # manifold argments
    parser.add_argument("--num_geod", default=5, type=int)  # num of random geodesics to plot
    parser.add_argument("--iter_energy", default=50, type=int)  # num of steps to minimise energy func
    # data used
    parser.add_argument("--data", default="font", type=str)
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--model_folder", default="models/font/", type=str)
    parser.add_argument("--exp_folder", default="plots/fontdata/", type=str)
    parser.add_argument("--model_title", default="model_1_f", type=str)
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))

    folderpath = os.path.abspath(opts.exp_folder)
    create_folder(folderpath)
    print("--- figures saved in:", folderpath)

    # load previously training gplvm model
    model_saved = pickle_load(folder_path=f"{opts.model_folder}", file_name=f"{opts.model_title}.pkl")
    model = model_saved["model"].base_model
    Y = model.y.data.numpy().transpose()
    X = model.X.data.numpy()

    # get Riemannian and Finslerian metric
    gplvm_riemann = Gplvm(model, mode="riemannian")
    gplvm_finsler = Gplvm(model, mode="finslerian")

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

    x0_index = np.argmax(np.linalg.norm(X, axis=1))
    x0 = np.tile(X[x0_index], (opts.num_geod, 1))
    x1 = X[np.random.choice(X.shape[0], opts.num_geod, replace=False)]

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

    colors = sns.color_palette("viridis", n_colors=opts.num_geod)
    # plot latent space with geodesics
    fig = plt.figure(1)
    ax = plt.axes()
    ax, im, hm_values, hm_ltt = volume_heatmap(ax, gplvm_riemann, X, mode="vol_riemann", n_grid=10)
    for i in range(opts.num_geod):
        ax.plot(c_coords_riemann[i, ::2, 0], c_coords_riemann[i, ::2, 1], c=colors[i], lw=1, alpha=0.5)
        ax.plot(c_coords_finsler[i, 1::2, 0], c_coords_finsler[i, 1::2, 1], c=colors[i], lw=1, ls=":")
    ax.scatter(X[:, 0], X[:, 1], marker="o", edgecolors="black", s=2)
    fig.colorbar(im)
    plt.title("Geodesic in the latent space")
    plt.show()

    filename = "latent" + "_" + opts.model_title + ".png"
    filepath = create_filepath(folderpath, filename)
    fig.savefig(filepath, dpi=fig.dpi)
    print("--- plot of the latent space saved as: {}".format(filepath))

    # plot font data space with geodesics
    fig = plt.figure(2)
    num_cols, num_rows = 12, opts.num_geod
    gs = fig.add_gridspec(num_rows, num_cols, hspace=0, wspace=0)  # 10 images along num_geod geodesics
    letters = c_obs_riemann.reshape((num_rows, eval_grid, 4, -1))

    pts_sampling = int(eval_grid / num_cols) - 1
    letters = letters[:, ::pts_sampling, :, :]
    letters = letters[:, :num_cols, :, :]

    axes = gs.subplots(sharex="col", sharey="row")
    for row in range(num_rows):
        for col in range(num_cols):
            axes[row, col].plot(letters[row, col, 0, :], letters[row, col, 2, :], color=colors[row])
            axes[row, col].plot(letters[row, col, 1, :], letters[row, col, 3, :], color=colors[row])
            axes[row, col].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            axes[row, col].set_aspect("equal")
            axes[row, col].axis("off")
    plt.show()

    filename = "observational" + "_" + opts.model_title + ".png"
    filepath = create_filepath(folderpath, filename)
    fig.savefig(filepath, dpi=fig.dpi)
    print("--- plot of the observational space saved as: {}".format(filepath))
