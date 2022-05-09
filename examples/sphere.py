import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pyro.contrib.gp as gp
import pyro.ops.stats as stats
import torch
from stochman.curves import CubicSpline
from stochman.geodesic import geodesic_minimizing_energy

from finsler.gplvm import gplvm
from finsler.utils.data import make_sphere_surface
from finsler.utils.helper import pickle_load, pickle_save
from finsler.utils.pyro import initialise_kernel, iteration


def get_args():
    parser = argparse.ArgumentParser()
    # manifold argments
    parser.add_argument("--n_samples", default=400, type=int)
    parser.add_argument("--latent_dim", default=2, type=int)
    parser.add_argument("--num_geod", default=8, type=int)  # num of random geodesics to plot
    # initialisation
    parser.add_argument("--init", default="iso", type=str)  # pca or iso
    parser.add_argument("--data", default="sphere", type=str)  # sphere or qPCR or font
    # kernel argments
    parser.add_argument("--lengthscale", default=1.0, type=float)
    parser.add_argument("--variance", default=0.1, type=float)
    parser.add_argument("--noise", default=0.01, type=float)
    # optimisation arguments
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--iter", default=6000, type=int)
    # load previous exp
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--exp_folder", default="trained_models/", type=str)
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))
    exp_folder = os.path.join(opts.exp_folder, opts.data)
    print("figures saved in:", exp_folder)

    if not opts.train:
        # load previously training gplvm model
        model_saved = pickle_load(folder_path=exp_folder, file_name="model_{}.pkl".format(opts.data))
        model = model_saved["model"]
        Y = model_saved["Y"]
        X = model_saved["X"]

    if opts.train:
        # initialise kernel
        Y, X, lengthscale, variance, noise = initialise_kernel(opts.data)
        kernel = gp.kernels.RBF(input_dim=opts.latent_dim, variance=variance, lengthscale=lengthscale)

        # Construct and train GP using pyro
        if opts.n_samples < 16:
            Xu = stats.resample(X.clone(), opts.n_samples)
        else:
            Xu = stats.resample(X.clone(), 16)
        gpmodule = gp.models.SparseGPRegression(X, Y.t(), kernel, Xu, noise=noise, jitter=1e-5)
        model = gp.models.GPLVM(gpmodule)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
        losses = iteration(model, optimizer, opts.iter, opts.n_samples)
        # plt.plot(losses); plt.xlabel('Iteration'); plt.ylabel('Loss (using ELBO)'); plt.show()

        # save model
        model_save = {"model": model, "kernel": kernel, "Y": Y, "X": X}
        pickle_save(model_save, exp_folder, file_name="model_{}.pkl".format(opts.data))

    gplvm_riemann = gplvm(model, mode="riemannian")
    gplvm_finsler = gplvm(model, mode="finslerian")
    X = model.X_loc.detach().numpy()

    # random points from the latent space
    x_random = np.random.uniform(low=-2, high=2, size=(100, 2))
    x_random = torch.tensor(x_random, dtype=torch.float32)
    y_random = gplvm_riemann.embed(x_random)[0].detach().numpy()  # y_random_mean

    # points defining the geodesic
    xx = torch.tensor([X[1], X[7]], dtype=torch.float32).reshape((2, 2))
    yy = gplvm_riemann.embed(xx.squeeze())[0].detach().numpy()

    # Energy function computed with riemannian metric
    c_coords_riemann, c_obs_riemann = torch.empty((opts.num_geod, 50, 2)), np.empty((opts.num_geod, 50, 3))
    c_coords_finsler, c_obs_finsler = torch.empty((opts.num_geod, 50, 2)), np.empty((opts.num_geod, 50, 3))
    for i in range(opts.num_geod):
        x_low, x_high = 0.9 * np.amin(X, axis=0)[1], 0.9 * np.amax(X, axis=0)[1]
        x_left, x_right = 0.7 * np.amin(X, axis=0)[0], 0.7 * np.amax(X, axis=0)[0]
        x_1q, x_3q = (3 * x_left + x_right) / 4, (x_left + 3 * x_right) / 4
        x0 = [np.random.uniform(x_left, x_1q, 1), np.random.uniform(x_low, x_high, 1)]
        x1 = [np.random.uniform(x_3q, x_right, 1), np.random.uniform(x_low, x_high, 1)]
        xx = torch.tensor([x0, x1], dtype=torch.float32).reshape((2, 2))

        print("\n")
        print("Riemannian")
        spline = CubicSpline(begin=xx[0].reshape(1, -1), end=xx[1].reshape(1, -1), requires_grad=True)
        result = geodesic_minimizing_energy(spline, gplvm_riemann, max_iter=15, eval_grid=20)
        c_coords_riemann[i, :, :] = spline(torch.linspace(0, 1, 50)).detach()

        print("\n")
        print("Finslerian")
        spline_finsler = CubicSpline(begin=xx[0].reshape(1, -1), end=xx[1].reshape(1, -1), requires_grad=True)
        result_finsler = geodesic_minimizing_energy(
            spline_finsler, gplvm_finsler, max_iter=20, eval_grid=20
        )  # eval_grid<20 to avoid numerical problems
        c_coords_finsler[i, :, :] = spline_finsler(torch.linspace(0, 1, 50)).detach()

    # plot observational space with geodesics
    fig = plt.figure(1)
    ax = plt.axes(projection="3d")
    XS, YS, ZS = make_sphere_surface(opts.n_samples)  # for illustration
    ax.plot_surface(XS, YS, ZS, alpha=0.2, shade=True, rstride=2, cstride=2, linewidth=0, color="gray")
    ax.plot_wireframe(XS, YS, ZS, rstride=2, cstride=2, linewidth=0.4, color="gray")
    ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2], s=2, label="observed data", zorder=0)  # observed data points
    ax.scatter3D(yy[:, 0], yy[:, 1], yy[:, 2], c="red", s=5)  # points of beginning and end of geodesic
    i = 0
    ax.scatter3D(c_obs_riemann[i, ::2, 0], c_obs_riemann[i, ::2, 1], c="orange", s=2, label="geodesic Riemann")
    ax.scatter3D(c_obs_finsler[i, 1::2, 0], c_obs_finsler[i, 1::2, 1], c="red", s=2, label="geodesic Finsler")
    ax.scatter3D(
        y_random[:, 0], y_random[:, 1], y_random[:, 2], c="green", s=5, label="random latent data"
    )  # random points taken from the latent
    ax.legend()
    ax.set_xlim((-1, 1)), ax.set_ylim((-1, 1)), ax.set_zlim((-1, 1))
    ax.grid(False)
    ax.axis("off")
    plt.show()
    fig.savefig(os.path.join(exp_folder, "sphere_observational.png"), dpi=fig.dpi)
