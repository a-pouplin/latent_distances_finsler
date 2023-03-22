# Gaussian process regression and computation of the gradient of the predictions.
# Example used to illustrate "evaluateDiffKernel".
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
from torch.autograd import Function, functional, gradcheck

np.random.seed(seed=42)


def to_np(x):
    return x.detach().numpy()


def to_torch(x):
    return torch.from_numpy(x)


def pairwise_distances(x, y):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def evaluateKernel(xstar, x):
    """
    For the RBF kernel, evaluate the kernel using the pairwise distances
    Returns the evaluated kernel of the latent variables and the new point
    """
    r = pairwise_distances(xstar, x)
    var, l = 1.0, 1.0
    ksx = var * torch.exp((-0.5 * r / l**2))
    return ksx


def embed(X, Y, xstar):
    """Maps from latent to data space, implements equations 2.23 - 2.24 from Rasmussen.
    We assume a different mean function across dimensions but same covariance matrix
    """
    n = X.shape[0]
    noise = 0  # noise = torch.randn(1).exp()**2
    X, Y = X.to(torch.float64), Y.to(torch.float64)
    xstar = xstar.to(torch.float64)  # Numerical precision needed

    Ksx = evaluateKernel(xstar, X)
    Kxx = evaluateKernel(X, X) + torch.eye(n) * noise
    Kss = evaluateKernel(xstar, xstar)
    Kinv = (Kxx).cholesky().cholesky_inverse()  # Kinv = Kxx.inverse()

    mu = Ksx.mm(Kinv).mm(Y)
    sigma = Kss - Ksx.mm(Kinv).mm(Ksx.T)  # should be symm. positive definite
    return mu, sigma


def evaluateDiffKernel(xstar, X):
    # compute the gradient of the kernel with gardient support
    N_train, d = X.shape
    kernel_jac = functional.jacobian(evaluateKernel, (xstar, X), create_graph=True)
    kernel_hes = functional.hessian(evaluateKernel, (xstar, xstar), create_graph=True)
    dK = torch.reshape(torch.squeeze(kernel_jac[0]), (N_train, d)).T
    ddK = torch.reshape(torch.squeeze(kernel_hes[0][1]), (d, d))
    return dK, ddK


def derivative_predictions(Y, X_train, X_test):
    # Compute the mean and variance of the derivatives
    # E. Solak (2002) "Derivative observations in Gaussian process models of dynamic systems"
    Y = Y.to(torch.float64)
    X_train = X_train.to(torch.float64)
    X_test = X_test.to(torch.float64)
    N_train, N_test, D = X_train.shape[0], X_test.shape[0], X_train.shape[1]
    noise = 0  # noise = (torch.randn(1).exp()**2)

    Kxx = evaluateKernel(X_train, X_train) + torch.eye(N_train) * noise
    Kinv = (Kxx).cholesky().cholesky_inverse()
    jac_mu = torch.zeros(N_test, D)  # mean of the derivatives Y
    jac_var = torch.zeros(N_test, D, D)  # variance of the derivatives of Y

    for nn in range(N_test):
        x_test = X_test[nn, :].unsqueeze(0)
        dK, ddK = evaluateDiffKernel(x_test, X)
        jac_var[nn, :, :] = ddK - dK.mm(Kinv).mm(dK.T)
        for dd in range(D):
            y = Y[:, dd].unsqueeze(1)
            jac_mu[nn, dd] = dK.mm(Kinv).mm(y)
    return jac_mu, jac_var


def plot_observations(Y, X, xstar, function):
    xstar = to_torch(np.sort(to_np(xstar), axis=0))
    ystar_mu, ystar_var = embed(X, Y, xstar)
    jac_mu, _ = derivative_predictions(Y, X, xstar)
    # from torch to numpy
    X = np.squeeze(to_np(X))
    Y = np.squeeze(to_np(Y))
    ystar_mu = np.squeeze(to_np(ystar_mu))
    ystar_var = np.squeeze(to_np(ystar_var.diag()))
    xstar = np.squeeze(to_np(xstar))
    jac_mu = np.squeeze(to_np(jac_mu))
    # plot figure with derivatives
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(np.linspace(X.min(), X.max(), 1000), function(np.linspace(X.min(), X.max(), 1000)), color="lightsteelblue")
    ax.plot(X, Y, "o", markersize=4, label="observations", color="royalblue")
    ax.plot(xstar, ystar_mu, "o", markersize=2, label="predictions", color="orange")
    # derivation represented by small segments
    def slop(x, y, dy, xrange):
        return dy * (xrange - x) + y

    for i in range(xstar.shape[0]):
        if i % 10 == 0:
            xs, ys, dys = xstar[i], ystar_mu[i], jac_mu[i]
            xrange = np.linspace(xs - 0.2, xs + 0.2, 10)
            ax.plot(xrange, slop(xs, ys, dys, xrange), "-", markersize=1, color="sienna")
    ax.fill_between(
        xstar, ystar_mu - 1.96 * np.sqrt(ystar_var), ystar_mu + 1.96 * np.sqrt(ystar_var), color="papayawhip", alpha=0.5
    )
    ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    function = lambda x: x * np.cos(x)
    X = torch.tensor(np.expand_dims(np.linspace(-10, 10, 50), axis=1), requires_grad=True)
    Y = torch.from_numpy(function(X.detach().numpy()))
    xstar = torch.tensor(np.expand_dims(np.random.uniform(-10, 10, 10), axis=1), requires_grad=True)
    plot_observations(Y, X, xstar, function)
    raise
