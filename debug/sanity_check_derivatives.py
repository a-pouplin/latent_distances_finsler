import os

import matplotlib.pyplot as plt
import numpy as np
import pyro.contrib.gp as gp
import torch
from torch.autograd import functional

from finsler.utils.helper import pickle_load, pickle_save, to_np, to_torch
from finsler.utils.pyro import iteration


def derivatives(coords, model, method="obs_derivatives"):
    """
    Function using two methods to obtain the variance and expectation of
    the partial derivatives (df/dt) of the map f.
    df/dt = df/dc * dc/dt.
    inputs:
        - coords: coords of latent variables from the spline (N_test points)
        - gpr: gaussian process
        - method: method used (discretization or observational derivatives)
    output:
        - var_derivatives: variance (vector, size: (N_test-1))
        - mu_derivatives: expectation (matrix, size: (N_test-1)*D)
    """
    gpr, X, Y = model["model"], model["X"], model["Y"]
    D, N = Y.shape
    Ntest, q = coords.shape

    mu_derivatives = torch.zeros(Ntest - 1, D, dtype=torch.float64)
    var_derivatives = torch.zeros(Ntest - 1)

    mu_star = torch.zeros(Ntest - 1, q, D)
    var_star = torch.zeros(Ntest - 1, q, q)

    X, Y = X.to(torch.float64), Y.to(torch.float64)  # num. precision

    if method == "discretization":
        mean_c, var_c = gpr.forward(coords, full_cov=True)
        mean_c = mean_c.T
        var_c = var_c[0]
        mu_derivatives = mean_c[1:, :] - mean_c[0:-1, :]
        var_derivatives = var_c.diagonal()[1:] + var_c.diagonal()[0:-1] - 2 * (var_c[0:-1, 1:].diagonal())

    elif method == "obs_derivatives":
        dc = coords[1:, :] - coords[0:-1, :]  # derivatives of the spline (dc/dt)
        c = coords[0:-1, :]  # coordinates of derivatives

        kxx = gpr.kernel.forward(X, X) + torch.eye(N) * gpr.noise  # (N x N)
        kinv = torch.cholesky_inverse(torch.linalg.cholesky(kxx))  # (N x N)

        for nn in range(Ntest - 1):
            dk, ddk = derivative_kernel(c[nn, :].unsqueeze(0), X, gpr)  # dk (N x q), ddk (1 x q x q)
            var_star[nn, :, :] = ddk - dk.mm(kinv).mm(dk.T)  # (1 x q x q)
            for dd in range(D):
                y = Y[dd, :].unsqueeze(1)  # (N x 1)
                mu_star[nn, :, dd] = dk.mm(kinv).mm(y)  # (q x N) x (N x N) x (N x 1)

        mu_derivatives = torch.einsum("bij, bi -> bj", mu_star, dc)
        var_dc = torch.einsum("bij, bi -> bj", var_star, dc)
        var_derivatives = torch.einsum("bi, bi -> b", dc, var_dc)

    return mu_derivatives, var_derivatives


def derivative_kernel(xstar, X, gpr):
    # Compute the differentiation of the kernel at a single point
    N, d = X.shape
    kernel_jac = functional.jacobian(gpr.kernel.forward, (xstar, X), create_graph=False)
    kernel_hes = functional.hessian(gpr.kernel.forward, (xstar, xstar), create_graph=False)
    dk = torch.reshape(torch.squeeze(kernel_jac[0]), (N, d)).T
    ddk = torch.reshape(torch.squeeze(kernel_hes[0][1]), (d, d))
    return dk, ddk


def initialise_kernel(num_points=50):
    dummy_func = lambda x: (np.cos(3 * x), np.sin(2 * x))
    X = torch.tensor(np.expand_dims(np.linspace(-np.pi, np.pi, num_points), axis=1), requires_grad=True)
    Y0, Y1 = dummy_func(X.detach().numpy())
    Y = torch.tensor(np.concatenate((Y0, Y1), axis=1))
    lengthscale = torch.tensor(0.1, requires_grad=True)
    variance = torch.tensor(10.0, requires_grad=True)
    noise = torch.tensor(0.001, requires_grad=False)
    kernel = gp.kernels.RBF(X.shape[1], variance, lengthscale)
    gpmodel = gp.models.GPRegression(X, Y.T, kernel, noise)
    return gpmodel


def train_gp(file_name):
    gpmodel = initialise_kernel()
    X, Y, kernel = gpmodel.X, gpmodel.y, gpmodel.kernel
    # optimizer
    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.01)
    losses = iteration(gpmodel, optimizer, 500, X.shape[0])
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (using ELBO)")
    plt.show()

    # save model
    model_save = {"model": gpmodel, "kernel": kernel, "Y": Y, "X": X}
    exp_folder = "trained_models/debug/"
    pickle_save(model_save, exp_folder, file_name)


def plot_observations(ax, model, xstar):
    # getting variables
    dummy_func = lambda x: (np.cos(3 * x), np.sin(2 * x))
    gpr, X, Y = model["model"], model["X"], model["Y"]
    xstar = to_torch(np.sort(xstar.detach().numpy(), axis=0))
    ystar_mu, ystar_var = gpr.forward(xstar, full_cov=False)
    # from torch to numpy
    X, Y = to_np(X), to_np(Y)
    ystar_mu, ystar_var = to_np(ystar_mu), to_np(ystar_var)
    xstar = to_np(xstar)
    # plot
    Y0, Y1 = dummy_func(np.linspace(X.min(), X.max(), 1000))
    ax.plot(Y0, Y1, color="lightsteelblue")
    # ax.plot(Y[0,:], Y[1,:], 'o', markersize=4, label='observations', color='royalblue')
    ax.plot(ystar_mu[0, ::5], ystar_mu[1, ::5], "o", markersize=2, label="predictions", color="orange")
    # ax.fill_between(xstar, ystar_mu - 1.96 * np.sqrt(ystar_var),
    #                        ystar_mu + 1.96 * np.sqrt(ystar_var),
    #                        color='papayawhip', alpha=0.5)
    return ax


def plot_derivatives_theory(ax, xstar):
    dummy_func = lambda x: (np.cos(3 * x), np.sin(2 * x))
    dummy_der = lambda x: (-3 * np.sin(3 * x), 2 * np.cos(2 * x))

    Y0, Y1 = dummy_func(xstar[::5])
    V0, V1 = dummy_der(xstar[::5])
    ax.quiver(Y0, Y1, V0, V1, label="theoretical derivatives", color="crimson", alpha=0.4, width=8e-3, zorder=2)
    return ax


def plot_derivatives(ax, model, xstar, method="obs_derivatives", label="Computing posterior", color="purple"):
    gpr, X = model["model"], to_np(model["X"])
    ystar_mu, _ = gpr.forward(xstar, full_cov=False)
    jac_mu, var_mu = derivatives(xstar, model, method)
    # torch to numpy
    ys, dys = to_np(ystar_mu[:, :-1]), to_np(jac_mu)
    ax.quiver(
        ys[0, ::5], ys[1, ::5], dys[::5, 0], dys[::5, 1], label=label, color=color, alpha=0.8, width=3e-3, zorder=5
    )
    return ax


if __name__ == "__main__":
    np.random.seed(42)

    file_name = "parametric.pkl"
    # train_gp(file_name)
    exp_folder = "trained_models/debug/"
    model = pickle_load(exp_folder, file_name)
    gpr, X, Y, kernel = model["model"], model["X"], model["Y"], model["kernel"]
    xstar = torch.tensor(np.expand_dims(np.linspace(-np.pi, np.pi, 100), axis=1), requires_grad=False)

    outDir = "/Users/alpu/Desktop/Finsler_figures/plots/appendix"

    fig = plt.figure()
    ax = plt.axes()
    ax = plot_observations(ax, model, xstar)

    ax = plot_derivatives(ax, model, xstar, method="obs_derivatives", label="computing posterior", color="darkcyan")
    ax = plot_derivatives(ax, model, xstar, method="discretization", label="discretization", color="rebeccapurple")
    ax = plot_derivatives_theory(ax, xstar)
    ax.legend(loc="lower center", ncol=4)
    plt.show()
    fig.savefig(os.path.join(outDir, "{}.svg".format("derivatives")), dpi=fig.dpi, bbox_inches="tight")

    # print(derivatives(xstar, model, method="discretization")[1][:10])
    # print(derivatives(xstar, model, method="obs_derivatives")[1][:10])
