import numpy as np
import torch
from torch.distributions import Normal

from finsler.distributions import NonCentralNakagami
from finsler.gplvm import Gplvm
from finsler.utils.helper import pickle_load, pickle_save, to_np, to_torch


def test_distr():

    D = 3
    nsamples = 100000
    mus = torch.stack([torch.rand(1) for i in range(D)], axis=1)  # list of means
    var = torch.abs(torch.rand(1))  # isotropic variance
    print(mus.shape, var.shape)
    zs = [Normal(loc=mus[:, i], scale=var) for i in range(D)]  # list of normal distributions
    zs_sample = torch.stack([torch.squeeze(z.sample((nsamples,))) for z in zs])  # list of samples

    print(mus)
    print(torch.mean(zs_sample, axis=1))
    print(mus.shape, torch.mean(zs_sample, axis=1).shape)

    print(torch.abs(torch.mean(zs_sample, axis=1) - mus) / torch.abs(mus))

    print(torch.allclose(torch.mean(zs_sample, axis=1), mus, atol=0.0, rtol=0.05))

    print("TESTING WISHART DISTRIBUTION")

    xs = torch.stack([torch.squeeze(z.sample((nsamples,))) ** 2 for z in zs])
    x = torch.sum(xs, axis=0)  # Wishart W_1 distribution with nsamples
    y = torch.sqrt(x)  # Nakagami distribution with nsamples
    print(y.shape)

    print("nakagami")
    print(torch.mean(y), torch.std(y))
    # print(mus)
    ncnakagami = NonCentralNakagami(mus, var)
    print(ncnakagami.expectation().squeeze(), ncnakagami.variance().squeeze())

    print(torch.allclose(torch.mean(y), ncnakagami.expectation(), atol=0.0, rtol=0.2))
    print(torch.allclose(torch.std(y), torch.sqrt(ncnakagami.variance()), atol=0.0, rtol=0.2))


def test_derivatives(model, spline):

    # get Riemannian and Finslerian manifolds
    gplvm = Gplvm(model, mode="riemannian")
    num_points = 10

    ystar, _ = model.forward(spline, full_cov=False)

    # spline = torch.empty((num_points, 2))
    # spline[:, 0], spline[:, 1] = torch.linspace(0, 1, num_points), torch.linspace(0, 1, num_points)

    """Test the derivatives of the Riemannian manifold"""
    deriv_discrete, _ = gplvm.derivatives(spline, method="discretization")
    deriv_diffgp, _ = gplvm.derivatives(spline, method="obs_derivatives")

    deriv_discrete = torch.squeeze(deriv_discrete)
    deriv_diffgp = torch.squeeze(deriv_diffgp)

    deriv_discrete = to_np(deriv_discrete)
    deriv_diffgp = to_np(deriv_diffgp)

    return deriv_diffgp, deriv_discrete, ystar


def plot_derivatives(ax, model, xstar, method="obs_derivatives", label="Computing posterior", color="purple", dim=2):
    gplvm = Gplvm(model, mode="riemannian")
    ystar_mu, _ = model.forward(xstar, full_cov=False)
    jac_mu, _ = gplvm.derivatives(xstar, method)
    # torch to numpy
    if dim == 2:
        ys, dys = to_np(ystar_mu[:, :-1]), to_np(jac_mu)
        ax.quiver(
            ys[0, ::5], ys[1, ::5], dys[::5, 0], dys[::5, 1], label=label, color=color, alpha=0.8, width=3e-3, zorder=5
        )
    elif dim == 3:
        ys, dys = to_np(ystar_mu[:, :-1]), to_np(jac_mu)
        ax.quiver(
            ys[0, ::5],
            ys[1, ::5],
            ys[2, ::5],
            dys[::5, 0],
            dys[::5, 1],
            dys[::5, 2],
            label=label,
            color=color,
            alpha=0.8,
            zorder=5,
        )
    return ax


if __name__ == "__main__":
    num_points = 50

    # load previously training gplvm model
    file_name = "parametric.pkl"
    exp_folder = "models/debug/"
    model_debug = pickle_load(exp_folder, file_name)
    model_debug = model_debug["model"]
    spline_debug = torch.tensor(np.expand_dims(np.linspace(-np.pi, np.pi, num_points), axis=1), requires_grad=False)
    print(model_debug)
    # gpr, X, Y, kernel = model["model"], model["X"], model["Y"], model["kernel"]

    file_name = "model.pkl"
    exp_folder = "models/qPCR/"
    # exp_folder = "models/starfish/"
    model = pickle_load(exp_folder, file_name)
    model = model["model"].base_model
    spline_starfish = torch.empty((num_points, 2))
    spline_starfish[:, 0], spline_starfish[:, 1] = torch.linspace(0, 1, num_points), torch.linspace(0, 1, num_points)
    print(model)
    # gpr, X, Y, kernel = model["model"], model["X"], model["Y"], model["kernel"]

    # deriv_diffgp, deriv_discrete, ystar = test_derivatives(model_debug, spline_debug)
    # plot the splines and the derivatives
    # import matplotlib.pyplot as plt

    # # model, xstar = model_debug, spline_debug
    model, xstar = model, spline_starfish
    print(xstar.shape)

    # ystar_mu, _ = model.embed(xstar, full_cov=False)
    # dim = ystar_mu.shape[0]
    # print('Print in {} dimensions'.format(dim))

    # if dim ==2:
    #     ax = plt.figure()
    #     ax.plot(to_np(ystar_mu[0]), to_np(ystar_mu[1]), "o", markersize=2, label="predictions", color="orange")
    # elif dim == 3:
    #     ax = plt.figure().add_subplot(projection='3d')
    #     ax.scatter(to_np(ystar_mu[0]), to_np(ystar_mu[1]),  to_np(ystar_mu[2]), "o", label="predictions", color="orange")
    # ax = plot_derivatives(ax, model, xstar, dim = dim, method="obs_derivatives", label="computing posterior", color="darkcyan")
    # ax = plot_derivatives(ax, model, xstar, dim = dim, method="discretization", label="discretization", color="rebeccapurple")
    # ax.legend(loc="lower center", ncol=4)
    # plt.show()

    gplvm = Gplvm(model, mode="finslerian")
    dmu, dvar = gplvm.derivatives(xstar)
    mean_c, var_c = gplvm.embed(xstar)
    ncn = NonCentralNakagami(dmu, dvar)

    D = mean_c.shape[0]

    variance_term = (2 * (var_c.trace() - var_c[1:, 0:-1].trace()) - var_c[0, 0] - var_c[-1, -1]).sum()
    mean_term = (mean_c[1:, :] - mean_c[0:-1, :]).pow(2).sum()

    energy1 = ((dmu.T) @ dmu).trace() + D * (dvar).sum()
    energy2 = mean_term + D * variance_term
    energy3 = (torch.nan_to_num(ncn.expectation(), nan=0) ** 2 + torch.nan_to_num(ncn.variance(), posinf=0)).sum()

    print(ncn.expectation(), ncn.variance())
    print(energy1)
    print(energy2)
    print(energy3)
    print((energy1 - energy3) / energy3)
    print(torch.allclose(energy1, energy3, atol=0, rtol=0.1))
