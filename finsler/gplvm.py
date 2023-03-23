import numpy as np
import torch
from pyro.contrib.gp.models import SparseGPRegression
from stochman.curves import CubicSpline
from stochman.manifold import Manifold
from torch.autograd import functional

from finsler.distributions import NonCentralNakagami
from finsler.kernels.rbf import RBF
from finsler.sasgp import SASGP


class Gplvm(Manifold):
    """Class that takes a gplvm model as input and allows for computation of curve energy"""

    def __init__(self, object, device=None, mode="riemannian"):
        self.model = object.to(device)  #
        self.device = device
        self.mode = mode  # Riemannian or Finslerian
        self.data = object.y

    def pairwise_distances(self, x, y=None):
        """
        Compute the pairwise distance matrix between two collections of vectors.
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def evaluateDiffKernel(self, xstar, X, kernel_type=None):
        """
        Compute the differentiation of the kernel.

        inputs:
            - xstar: data to be predicted (size: 1xd)
            - X: training data (size: Nxd)
        outputs:
            - dK: first derivative of the kernel (matrix, size: dxN)
            - ddK: second derivative of the kernel (matrix, size: dxd)
        """
        N_train, d = X.shape
        if isinstance(self.model.kernel, RBF) or kernel_type == "rbf":  # faster for RBF
            if isinstance(self.model, SparseGPRegression):  # work with a pyro module
                radius = self.pairwise_distances(xstar, X)
                var = self.model.kernel.variance_unconstrained.exp()
                length = self.model.kernel.lengthscale_unconstrained.exp()
                ksx = var * torch.exp((-0.5 * radius / length**2))
                dK = -(length**-2) * (xstar - X).T * ksx  # (d, N)
                ddK = (length**-2 * var) * torch.eye(d).to(torch.float64)  # (d, d)

            elif isinstance(self.model, SASGP):  # work with a SAS module
                assert xstar.dim() == 2, "cannot work with batches yet "
                xstar = xstar.unsqueeze(0)  # (1, num_x, num_dim)
                X = torch.unsqueeze(self.model.X, 0)  # (1, num_data, num_dim)
                lengthscale = self.model.kernel.length_scale.unsqueeze(0)  # (1, 2)
                var = self.model.kernel.variance  # (1, 1)
                Ksx = self.model.kernel.K(xstar, X)  # (1, num_x, num_data)

                # dK = -2 * lengthscale * (xstar - X).T * Ksx
                Xminusxstar = torch.cat(
                    [xstar[:, i, :] - X for i in range(xstar.shape[1])]
                )  # (num_x, num_data, num_dim)
                Xminusxstar = Xminusxstar.transpose(1, 2)  # (num_x, num_dim, num_data)
                Ksx = Ksx.repeat(xstar.shape[1], 1, 1)  # (num_x, num_x, num_data)
                lengthscale = lengthscale.repeat(xstar.shape[1], 1, 1)  # (num_x, 1, 2)
                ## (num_x, num_data, num_dim) = (num_x, 1, num_data) * (num_x, num_dim, num_data)
                dk = -2 * torch.bmm(Xminusxstar, Ksx)  # (num_x, num_data, num_dim)
                ddK = 2 * lengthscale**2 * torch.eye(d).to(torch.float64)  # (num_x, num_dim, num_dim)
                raise NotImplementedError
        else:
            kernel_jac = functional.jacobian(self.model.kernel.forward, (xstar, X), create_graph=False)
            kernel_hes = functional.hessian(self.model.kernel.forward, (xstar, xstar), create_graph=False)
            dK = torch.reshape(torch.squeeze(kernel_jac[0]), (N_train, d)).T
            ddK = torch.reshape(torch.squeeze(kernel_hes[0][1]), (d, d))
        return dK, ddK

    def embed(self, xstar, full_cov=True):
        if isinstance(self.model, SparseGPRegression):  # work with a pyro module
            loc, cov = self.model.forward(xstar, full_cov=full_cov)
        elif isinstance(self.model, SASGP):  # work with a SAS module
            loc, cov = self._embed_sas(xstar, full_cov=full_cov)
        # loc, cov = self.model.forward(torch.squeeze(xstar).float(), full_cov=full_cov)
        return loc, cov  # dims: D x Nstar, Nstar x Nstar

    def _embed_sas(self, xstar, full_cov):
        # Maps from latent to data space, implements equations 2.23 - 2.24 from Rasmussen.
        assert isinstance(self.model.kernel, RBF), "embed only works with SASGP model"
        assert hasattr(self.model, "Kinv"), "Kinv does not exist"  # assert that Kinv exists

        # add batch dimension
        if xstar.dim() == 2:
            xstar = xstar.unsqueeze(0)
        X = torch.unsqueeze(self.model.X, 0)
        Y = torch.unsqueeze(self.model.y, 0)

        # shapes
        (bs, num_data, _) = xstar.shape  # (batch size, number of data points, latent dim == 2))
        obs_dim = Y.shape[-1]  # observational dimension (= 784 if MNIST)

        Ksx = self.model.kernel.K(xstar, X)
        Kss = self.model.kernel.K(xstar, xstar)
        Kinv = self.model.Kinv  # hard coded for now

        # check shapes and ensure batch computations
        assert Kinv.dim() == 3, "Kinv should be a 3D tensor"
        assert Kss.dim() == 3, "Kss should be a 3D tensor"
        assert Ksx.dim() == 3, "Ksx should be a 3D tensor"
        Kinv = Kinv.repeat(bs, 1, 1)  # repeat Kinv for each batch
        Y = Y.repeat(bs, 1, 1)  # repeat Kss for each batch

        mu = torch.bmm(Ksx, torch.bmm(Kinv, Y))  # shape: (batch size, number of test points)
        Sigma = Kss - torch.bmm(
            Ksx, torch.bmm(Kinv, Ksx.transpose(-2, -1))
        )  # shape: (batch size, number of test points, number of test points)

        # check shapes
        assert mu.shape == (bs, num_data, obs_dim), "mu has wrong shape"
        assert Sigma.shape == (bs, num_data, num_data), "Sigma has wrong shape"
        L = torch.linalg.cholesky_ex(Sigma)  # test if Sigma is positive definite

        if full_cov is False:
            Sigma = torch.diagonal(Sigma, dim1=1, dim2=2)
        return mu, Sigma

    def derivatives(self, coords, method="discretization"):
        """
        Function using two methods to obtain the variance and expectation of
        the partial derivatives (df/dt) of the map f.
        df/dt = df/dc * dc/dt.
        inputs:
            - coords: coords of latent variables from the spline (N_test points)
            - method: method used (discretization or observational derivatives)
        output:
            - var_derivatives: variance (vector, size: (N_test-1))
            - mu_derivatives: expectation (matrix, size: (N_test-1)*D)
        """
        X = self.model.X
        Y = self.model.y
        # noise = self.model.noise_unconstrained.exp() ** 2
        noise = 1e-8
        D, N = Y.shape
        Ntest, q = coords.shape

        mu_star = torch.zeros(Ntest - 1, q, D, dtype=torch.float64)
        var_star = torch.zeros(Ntest - 1, q, q, dtype=torch.float64)

        mu_derivatives = torch.zeros(Ntest - 1, D, dtype=torch.float64)
        var_derivatives = torch.zeros(Ntest - 1, dtype=torch.float64)

        if method == "discretization":
            # Needs a lot of samples but faster
            mean_c, var_c = self.embed(coords)
            mu_derivatives = mean_c[1:, :] - mean_c[0:-1, :]
            var_derivatives = var_c.diagonal()[1:] + var_c.diagonal()[0:-1] - 2 * (var_c[0:-1, 1:].diagonal())
            # mu, var = mu_{i+1} - mu_{i}, s_{i+1,i+1} + s_{i,i} - 2*s_{i,i+1}

        elif method == "obs_derivatives":
            # Get the CubicSpline class and get the derivatives
            dc = coords[1:, :] - coords[0:-1, :]  # derivatives of the spline (dc/dt)
            c = coords[0:-1, :]  # coordinates of derivatives
            X, Y = X.to(torch.float64), Y.to(torch.float64)
            c, dc = c.to(torch.float64), dc.to(torch.float64)

            kxx = self.model.kernel.forward(X, X) + torch.eye(N) * noise  # (N x N)
            kinv = torch.cholesky_inverse(torch.linalg.cholesky(kxx))  # (N x N)

            for nn in range(Ntest - 1):
                dk, ddk = self.evaluateDiffKernel(c[nn, :].unsqueeze(0), X)  # dk (N x q), ddk (1 x q x q)
                var_star[nn, :, :] = ddk - dk.mm(kinv).mm(dk.T)  # (1 x q x q)
                for dd in range(D):
                    y = Y[dd, :].unsqueeze(1)  # (N x 1)
                    mu_star[nn, :, dd] = (dk.mm(kinv).mm(y))[0]  # (q x N) x (N x N) x (N x 1)

            mu_derivatives = torch.einsum("bij, bi -> bj", mu_star, dc)
            var_dc = torch.einsum("bij, bi -> bj", var_star, dc)
            var_derivatives = torch.einsum("bi, bi -> b", dc, var_dc)

        return mu_derivatives, var_derivatives

    def jacobian_posterior(self, xstar):
        X = self.model.X.data
        Y = self.model.y.data
        noise = self.model.noise_unconstrained.data.exp() ** 2
        D, N = Y.shape
        Ntest, d = xstar.shape

        # enforce all variables to be double precision
        X, Y = X.to(torch.float64), Y.to(torch.float64)
        xstar = xstar.to(torch.float64)
        noise = noise.to(torch.float64)
        mu_star = torch.empty(Ntest, d, D)
        var_star = torch.empty(Ntest, d, d)

        kxx = self.model.kernel.forward(X) + torch.eye(N) * noise
        kinv = torch.cholesky_inverse(torch.linalg.cholesky(kxx))

        for nn in range(Ntest):
            dk, ddk = self.evaluateDiffKernel(xstar[nn, :].unsqueeze(0), X)
            var_star[nn, :, :] = ddk - dk.mm(kinv).mm(dk.T)  # var(df/dc)
            mu_star[nn, :, :] = dk.mm(kinv).mm(Y.T)  # mean(df/dc)
        return mu_star.to(torch.float32), var_star.to(torch.float32)

    def curve_energy(self, coords):
        D, n = self.data.shape
        dmu, dvar = self.derivatives(coords)
        if self.mode == "riemannian":
            energy = ((dmu.T) @ dmu).trace() + D * (dvar).sum()
        elif self.mode == "finslerian":
            non_central_nakagami = NonCentralNakagami(dmu, dvar)
            energy = (non_central_nakagami.expectation() ** 2).sum()
        print("{} energy: {:.4f} \r".format(self.mode, energy.detach().numpy()), end="\r")
        return energy

    def compute_derivatives_batch(self, curve: CubicSpline):
        """
        Computes the derivatives of the curve
        :param curve: (bs, num_data, dim_data)
        :return: (bs, num_data - 1, dim_data), (bs, num_data - 1)
        """
        assert curve.shape[1] >= 2, "Curves must have at least 2 points"
        loc, cov = self.embed(curve)  # loc (bs, num_data, dim_data), cov (bs, num_data, dim_data, dim_data)

        # Compute derivatives by discretising the curve step by step
        loc_der = loc[:, 1:, :] - loc[:, 0:-1, :]  # (bs, num_data - 1, dim_data)
        cov_nondiag_ = torch.diagonal(cov, offset=1, dim1=1, dim2=2)  # Upper diagonal of cov (bs, num_data - 1)
        cov_diag_ = torch.diagonal(cov, offset=0, dim1=1, dim2=2)  # diagonal of cov (bs, num_data - 1)
        cov_der = cov_diag_[:, 1:] + cov_diag_[:, 0:-1] - 2 * cov_nondiag_  # (bs, num_data - 1)

        # warnings if wrong shapes
        assert loc_der.shape == (
            curve.shape[0],
            curve.shape[1] - 1,
            loc.shape[-1],
        ), "local derivative shape: {}, expected {}".format(
            loc_der.shape, (curve.shape[0], curve.shape[1] - 1, loc.shape[-1])
        )
        assert cov_der.shape == (
            curve.shape[0],
            curve.shape[1] - 1,
        ), "covariant derivative shape: {}, expected {}".format(cov_der.shape, (curve.shape[0], curve.shape[1] - 1))
        return loc_der, cov_der

    def curve_energy_batch(self, coords: CubicSpline):
        """
        Computes the energy of the curve with batched computation
        :param coords: (bs, num_data, dim_data)
        :param mode: "riemannian" or "finslerian"
        :return: (bs)
        """
        loc_der, cov_der = self.compute_derivatives_batch(coords)  # (bs, num_data - 1, dim_data), (bs, num_data - 1)
        data_dim = self.model.y.shape[-1]  # observed data dimensions. For example, 784 for MNIST

        if self.mode == "riemannian":
            energy = (loc_der**2).sum(dim=-1).sum(dim=-1) + data_dim * cov_der.sum(dim=-1)
        elif self.mode == "finslerian":
            non_central_nakagami = NonCentralNakagami(loc_der, cov_der)
            energy = (non_central_nakagami.expectation() ** 2).sum(dim=-1)

        assert energy.shape == (coords.shape[0],), "Energy shape: {}, expected {}".format(
            energy.shape, (coords.shape[0],)
        )
        return energy

    def curve_length(self, curve: CubicSpline, dt=None):
        """
        Compute the discrete length of a given curve.

        Input:
            curve:      a Nx(d) torch Tensor representing a curve or
                        a BxNx(d) torch Tensor representing B curves.

        Output:
            length:     a scalar or a B element Tensor containing the length of
                        the curve.

        Algorithmic note:
            The default implementation of this function rely on the 'inner'
            function, which in turn call the 'metric' function. For some
            manifolds this can be done more efficiently, in which case it
            is recommended that the default implementation is replaced.
        """
        assert curve.shape[-1] == 2, "Should get latent dimensions: 2 but got {} instead".format(curve.shape[-1])

        if curve.dim() == 2:
            curve = torch.unsqueeze(curve, 0)  # add batch dimension if one isn't present
        if dt is None:
            dt = 1.0  # (curve.shape[1]-1)

        # Compute derivatives by discretising the curve step by step
        energy_batch = self.curve_energy_batch(curve)

        return torch.sqrt(energy_batch)

    def metric(self, xstar):
        """Computes the **expected** Riemannian metric at points xstar as the metric has to be deterministic
        - input: Points at which to compute the expected metric. Should be of size nof_points x d
        - output: Expected metric tensor of shape nof_points x d x d
        Ref: Tosi (2014), eq 22: https://arxiv.org/abs/1411.7432
        """
        J_mu, J_cov = self.jacobian_posterior(xstar)
        JJ_mu = torch.bmm(J_mu, J_mu.transpose(1, 2))
        D = self.data.shape[0]
        return D * J_cov + JJ_mu
