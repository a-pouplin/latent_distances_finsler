# latent space visualisations for a trained GPLVM model

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from stochman.curves import CubicSpline
from stochman.geodesic import geodesic_minimizing_energy, geodesic_minimizing_energy_sgd

from finsler.gplvm import gplvm
from finsler.utils.data import make_torus_surface, on_torus
from finsler.utils.helper import pickle_load
from finsler.visualisation.latent import volume_heatmap


def get_args():
    parser = argparse.ArgumentParser()
    # manifold argments
    parser.add_argument("--num_geod", default=5, type=int)  # num of random geodesics to plot
    parser.add_argument("--iter_energy", default=500, type=int)  # num of steps to minimise energy func
    # data used
    parser.add_argument("--data", default="torus", type=str)  # sphere or qPCR or font or starfish
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--model_folder", default="trained_models/", type=str)
    parser.add_argument("--exp_folder", default="plots/torus/", type=str)
    parser.add_argument("--title_model", default="", type=str)
    parser.add_argument("--final_plots", default="/Users/alpu/Desktop/Finsler_figures/plots", type=str)
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))
    exp_folder = opts.exp_folder
    print("figures saved in:", exp_folder)

    # load previously training gplvm model
    model_folder = os.path.join(opts.model_folder, opts.data)
    model_saved = pickle_load(folder_path=model_folder, file_name="model{}.pkl".format(opts.title_model))
    model = model_saved["model"]
    Y = model_saved["Y"]
    X = model_saved["X"]
    print(Y.shape)

    # get Riemannian and Finslerian metric
    gplvm_riemann = gplvm(model, mode="riemannian")
    gplvm_finsler = gplvm(model, mode="finslerian")
    X = model.X_loc.detach().numpy()

    # x_random = np.random.uniform(low=np.min(X), high=np.max(X), size=(100, 2))
    # x_random = torch.tensor(x_random, dtype=torch.float32)
    # y_random = gplvm_riemann.embed(x_random)[0].detach().numpy()  # y_random_mean
    # print('Percentage of latent points on torus: {:.2f}'.format(100*on_torus(y_random)))

    # raise
    # Energy function computed with riemannian metric
    optimizer = torch.optim.LBFGS
    eval_grid = 100
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
    X_m = X[np.linalg.norm(X - X[x0_index], axis=1) < 4]
    x1 = X_m[np.random.choice(X_m.shape[0], opts.num_geod, replace=False)]

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
    ax, im, hm_values, hm_ltt = volume_heatmap(ax, gplvm_riemann, X, mode="variance", n_grid=5)
    for i in range(opts.num_geod):
        ax.plot(c_coords_riemann[i, ::2, 0], c_coords_riemann[i, ::2, 1], c=colors[i], lw=1, alpha=0.5)
        ax.plot(c_coords_finsler[i, 1::2, 0], c_coords_finsler[i, 1::2, 1], c=colors[i], lw=1, ls=":")
    ax.scatter(X[:, 0], X[:, 1], marker="o", edgecolors="black", s=1)
    fig.colorbar(im)
    plt.title("Geodesic in the latent space")
    plt.show()
    fig.savefig(os.path.join(exp_folder, "latent{}.png".format(opts.title_model)), dpi=fig.dpi)
    fig.savefig(os.path.join(opts.final_plots, "latent_{}.svg".format(opts.data)))

    # plot observational space with geodesics
    fig = plt.figure(2)
    ax = plt.axes(projection="3d")
    ax.set_box_aspect((np.ptp(Y[:, 0]), np.ptp(Y[:, 1]), np.ptp(Y[:, 2])))
    XS, YS, ZS = make_torus_surface()  # for illustration
    ax.plot_surface(XS, YS, ZS, shade=True, color="gray", alpha=0.5, zorder=0)
    ax.scatter3D(
        Y[:500, 0], Y[:500, 1], Y[:500, 2], label="observed data", marker="o", edgecolors="black", s=1, zorder=2
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
    ax.grid(False)
    ax.axis("off")
    plt.show()
    fig.savefig(os.path.join(exp_folder, "observational.png"), dpi=fig.dpi)
    fig.savefig(os.path.join(opts.final_plots, "observational_{}.svg".format(opts.data)))
    # pickle_save(fig, opts.exp_folder,file_name='plot_test{}.pkl'.format(opts.data)) # use pickle to save the matplotlib object
