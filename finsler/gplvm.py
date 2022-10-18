import numpy as np
import torch
from stochman.curves import CubicSpline
from stochman.manifold import Manifold
from torch.autograd import functional

from finsler.distributions import NonCentralNakagami


def print_is_positive_definite(matrix):
    matrix = matrix.detach().numpy()
    print("Is positive definite symmetric: {}".format(np.all(np.linalg.eigvals(matrix) > 0)))


class gplvm(Manifold):
    """Class that takes a gplvm model as input and allows for computation of curve energy"""

    def __init__(self, object, device=None, mode="riemannian"):
        self.model = object.to(device)  #
        self.device = device
        self.mode = mode  # Riemannian or Finslerian
        self.data = object.base_model.y

    def pairwise_distances(self, x, y=None):
        """
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

    def evaluateKernel(self, xstar, x):
        """
        For the RBF kernel, evaluate the kernel using the pairwise distances
        Returns the evaluated kernel of the latent variables and the new point
        """
        radius = self.pairwise_distances(xstar, x)
        var = self.model.base_model.kernel.variance_unconstrained.exp()
        length = self.model.base_model.kernel.lengthscale_unconstrained.exp()

        ksx = var * torch.exp((-0.5 * radius / length**2))
        return ksx

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
        if kernel_type == "rbf":  # faster for RBF
            var = self.model.base_model.kernel.variance_unconstrained.exp()
            length = self.model.base_model.kernel.lengthscale_unconstrained.exp()
            ksx = self.evaluateKernel(xstar, X)
            dK = -(length**-2) * (xstar - X).T * ksx
            ddK = (length**-2 * var) * torch.eye(d).to(torch.float64)
        else:
            # kernel_jac = functional.jacobian(self.evaluateKernel, (xstar, X), create_graph=True)
            # kernel_hes = functional.hessian(self.evaluateKernel, (xstar, xstar), create_graph=True)
            kernel_jac = functional.jacobian(self.model.base_model.kernel.forward, (xstar, X), create_graph=False)
            kernel_hes = functional.hessian(self.model.base_model.kernel.forward, (xstar, xstar), create_graph=False)
            dK = torch.reshape(torch.squeeze(kernel_jac[0]), (N_train, d)).T
            ddK = torch.reshape(torch.squeeze(kernel_hes[0][1]), (d, d))
        return dK, ddK

    def embed_old(self, xstar):
        """Maps from latent to data space, implements equations 2.23 - 2.24 from Rasmussen.
        We assume a different mean function across dimensions but same covariance matrix
        """
        X = self.model.X_loc
        Y = self.model.base_model.__dict__["y"].clone().detach().t()
        n = X.shape[0]
        noise = self.model.base_model.noise_unconstrained.exp() ** 2
        X, Y = X.to(torch.float64), Y.to(torch.float64)
        xstar = xstar.to(torch.float64)  # Numerical precision needed

        Ksx = self.evaluateKernel(xstar, X)
        Kxx = self.evaluateKernel(X, X) + torch.eye(n) * noise
        Kss = self.evaluateKernel(xstar, xstar)
        Kinv = torch.cholesky_inverse(torch.linalg.cholesky(Kxx))
        # Kinv = (Kxx).cholesky().cholesky_inverse()  # Kinv = Kxx.inverse()

        mu = Ksx.mm(Kinv).mm(Y)
        Sigma = Kss - Ksx.mm(Kinv).mm(Ksx.T)  # should be symm. positive definite
        # print('MU:',mu[0,:].detach().numpy())
        # print('SIGMA:',Sigma[0,0].detach().numpy())
        return mu, Sigma

    def embed_old2(self, xstar, jitter=1e-5):
        """Maps from latent to data space, implements equations 2.23 - 2.24 from Rasmussen.
        We assume a different mean function across dimensions but same covariance matrix
        """
        X = self.model.X_loc
        Y = self.model.base_model.__dict__["y"].clone().detach().t()
        n = X.shape[0]
        noise = self.model.base_model.noise_unconstrained.exp() ** 2
        X, Y = X.to(torch.float64), Y.to(torch.float64)
        xstar = xstar.to(torch.float64)  # Numerical precision needed

        Ksx = self.model.base_model.kernel.forward(xstar, X)  # (Nstar,N)
        Kxx = self.model.base_model.kernel.forward(X, X) + torch.eye(n).to(self.device) * noise  # (N,N)
        Kss = self.model.base_model.kernel.forward(xstar, xstar)  # (Nstar,Nstar)
        Kinv = torch.cholesky_inverse(torch.linalg.cholesky(Kxx))  # (N,N)

        mu = Ksx.mm(Kinv).mm(Y)
        Sigma = Kss - Ksx.mm(Kinv).mm(Ksx.T) + jitter * torch.eye(xstar.shape[0])  # should be symm. positive definite
        return mu, Sigma

    def embed(self, xstar, full_cov=True):
        loc, cov = self.model.base_model.forward(torch.squeeze(xstar).float(), full_cov=full_cov, noiseless=False)

        return loc.T, cov[0]  # dims: D x Nstar, Nstar x Nstar

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
        X = self.model.X_loc
        Y = self.model.base_model.__dict__["y"].clone().detach()
        # noise = self.model.base_model.noise_unconstrained.exp() ** 2
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

            kxx = self.model.base_model.kernel.forward(X, X) + torch.eye(N) * noise  # (N x N)
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
        # TODO: make sure this is working to compute the energy for Riemann function
        X = self.model.X_loc
        Y = self.model.base_model.__dict__["y"].clone().detach()
        X, Y = X.to(torch.float64), Y.to(torch.float64)  # num. precision
        xstar = xstar.to(torch.float64)
        noise = self.model.base_model.noise_unconstrained.exp() ** 2
        D, N = Y.shape
        Ntest, d = xstar.shape

        mu_star, var_star = torch.empty(Ntest, d, D), torch.empty(Ntest, d, d)
        kxx = self.model.base_model.kernel.forward(X, X) + torch.eye(N) * noise
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
        if curve.dim() == 2:
            curve.unsqueeze_(0)  # add batch dimension if one isn't present
        if dt is None:
            dt = 1.0  # (curve.shape[1]-1)
        # Now curve is BxNx(d)
        emb_curve, _ = self.embed(curve)  # BxNxD
        emb_curve = emb_curve.unsqueeze_(0)
        delta = emb_curve[:, 1:] - emb_curve[:, :-1]  # Bx(N-1)xD
        speed = delta.norm(dim=2)  # Bx(N-1)
        lengths = speed.sum(dim=1) * dt  # B
        return lengths

    def evaluateDiffKernel_batch(self, xstar, X, kernel_type=None):
        # TODO: find a workaournd to get batch with
        # functional.jacobian and functional.hessian
        """
        Compute the differentiation of the kernel at a single point

        inputs:
            - xstar: data to be predicted (size: nof_points x d)
            - X: training data (size: Nxd)
        outputs:
            - dK: first derivative of the kernel (matrix, size: dxN)
            - ddK: second derivative of the kernel (matrix, size: dxd)
        """
        N_train, D = X.shape
        if kernel_type == "rbf":  # faster for RBF
            if len(xstar.shape) < 2:
                xstar = xstar.unsqueeze(0)
            nof_points = xstar.shape[0]

            var = self.model.base_model.kernel.variance_unconstrained.exp()
            length = self.model.base_model.kernel.lengthscale_unconstrained.exp()

            ksx = self.model.base_model.kernel.forward(xstar, X)

            X = torch.cat(nof_points * [X.unsqueeze(0)])
            xstar = torch.cat(N_train * [xstar.unsqueeze(0)]).transpose(0, 1)

            dK_term1 = xstar - X
            dK_term2 = torch.cat(D * [ksx.unsqueeze(0)]).transpose(0, 1).transpose(1, 2)

            dK = -(length**-2) * dK_term1 * dK_term2
            ddK = (length**-2 * var) * torch.eye(D).to(self.device)
        else:
            # TODO: @Alison: Let's use forward! I guess we could just parse that instead?
            # Eg:
            # kernel_jac = functional.jacobian(self.model.base_model.kernel.forward, (xstar, X), create_graph=True)
            # kernel_hes = functional.hessian(self.model.base_model.kernel.forward, (xstar, xstar), create_graph=True)
            kernel_jac = functional.jacobian(self.model.base_model.kernel.forward, (xstar, X), create_graph=False)
            kernel_hes = functional.hessian(self.model.base_model.kernel.forward, (xstar, xstar), create_graph=False)
            dK = torch.reshape(torch.squeeze(kernel_jac[0]), (N_train, D)).unsqueeze(0)
            ddK = torch.reshape(torch.squeeze(kernel_hes[0][1]), (D, D))
        return dK, ddK

    def jacobian(self, xstar):
        """
        Returns the expected Jacobian at xstar

        - input
            xstar   : Points at which to compute the expected metric.
                      Size: nof_points x d
        - output
            mus     : Mean of Jacobian distribution at xstar.
                      Size: nof_points x d x D
            covs    : Covariance of Jacobian distribution at xstar (assumed independent across dimensions).
                      Size: nof_points x d x d
        """
        try:
            x = self.model.X_loc
        except AttributeError:
            x = self.model.X_map

        sigma2 = self.model.base_model.noise_unconstrained.exp() ** 2
        if sigma2 < 1e-4:
            sigma2 = 1e-4
        N, d = x.shape
        Y = self.data  # .t()
        x, Y = x.to(torch.float64), Y.to(torch.float64)  # num. precision
        xstar = xstar.to(torch.float64)

        dk, ddk = self.evaluateDiffKernel_batch(xstar, x)
        kxx = self.model.base_model.kernel.forward(x, x) + torch.eye(N).to(self.device) * sigma2

        kinv = (kxx).cholesky().cholesky_inverse()

        mu_star = torch.matmul(dk.transpose(1, 2), kinv.mm(Y.t()))
        cov_star = ddk - torch.bmm(torch.matmul(dk.transpose(1, 2), kinv), dk)
        # if xstar.shape[0] == 1: # number of points
        #     mu_star = mu_star.squeeze()
        #     cov_star = cov_star.squeeze()
        return mu_star, cov_star

    def sample_jacobian(self, xstar, nof_samples=1):
        """
        Returns samples of Jacobian at point xstar.
        - input
            xstar           : Points...
            nof_samples     : Number of desired sampled at each point
        - output
            sample_jacobian : Size: nof_samples x nof_points x d x D.

        Note that looping is a lot faster than sampling once with a full covariance matrix!
        """

        def kronecker(A, B):
            return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0), A.size(1) * B.size(1))

        Y = self.model.base_model.__dict__["y"].clone().detach().t()
        D = Y.shape[1]

        if len(xstar.shape) < 2:
            xstar = xstar.unsqueeze(0)
        nof_points = xstar.shape[0]

        # mu_J,cov_J = self.jacobian(xstar)
        mu_J, cov_J = self.jacobian_posterior(xstar)

        j_samples_by_point = list()  # list of samples of each point, length: nof_samples
        for sample_index in range(nof_samples):
            j_by_point = list()  # list of j by point, length: nof_points
            for point_index in range(nof_points):
                mean = mu_J[point_index, :].view(1, -1).squeeze()
                cov = kronecker(cov_J[point_index, :], torch.eye(D))
                cov = torch.linalg.cholesky(cov) + torch.linalg.cholesky(cov.t(), upper=True)  # force symmetry

                # Haven't tested that the mean should be explicitly be converted to double
                # but I (cife) would think this is correct
                J = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
                # J = torch.distributions.multivariate_normal.MultivariateNormal(mean.double(),
                #                                               covariance_matrix=cov.double())
                sample = J.rsample()
                # TODO: Understand rsample vs sample. This has  something to do with pairwise derivatives...
                j_by_point.append(sample.reshape(mu_J.shape[1:]))
            j_samples_by_point.append(torch.stack(j_by_point).unsqueeze(0))
        sample_jacobian = torch.cat(j_samples_by_point)
        # if nof_points == 1:
        #     sample_jacobian = sample_jacobian.squeeze()
        # if nof_samples == 1:
        #     sample_jacobian = sample_jacobian.squeeze()
        return sample_jacobian

    def metric(self, xstar):
        """Computes the **expected** Riemannian metric at points xstar as the metric has to be deterministic
        - input: Points at which to compute the expected metric. Should be of size nof_points x d
        - output: Expected metric tensor of shape nof_points x d x d
        Ref: Tosi (2014), eq 22: https://arxiv.org/abs/1411.7432
        """
        J_mu, J_cov = self.jacobian_posterior(xstar)
        JJ_mu = torch.bmm(J_mu, J_mu.transpose(1, 2))
        D = self.data.shape[0]
        # J_cov = 0
        # G_expected = self.data.shape[0]*cov_J + torch.bmm(mu_Js,mu_Js.transpose(1,2))
        # G_expected = (1./self.data.shape[1])*torch.bmm(mu_Js,mu_Js.transpose(1,2)) + cov_J
        return D * J_cov + JJ_mu

    def sample_metric(self, xstar, nof_samples=50):
        """Allows sampling from the Riemannian metric at multiple points.
        - input:
            xstar: Points at which to compute the expected metric. Should be of size nof_points x d
            nof_samples: Number of desired samples
        - output:
            g_sample: Samples of the metric. Size (nof_points, nof_samples, d x d)
        """
        j_samples = self.sample_jacobian(xstar, nof_samples=nof_samples)
        g_sample = torch.matmul(j_samples, j_samples.transpose(2, 3))
        # return g_sample
        return torch.mean(g_sample, dim=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from finsler.utils.helper import pickle_load

    # load previously training gplvm model
    model_saved = pickle_load(folder_path="trained_models/concentric_circles", file_name="model_3_500.pkl")
    model = model_saved["model"]
    Y = model_saved["Y"]
    X = model_saved["X"]

    # get Riemannian and Finslerian manifolds
    gplvm_riemann = gplvm(model, mode="riemannian")
    gplvm_finsler = gplvm(model, mode="finslerian")

    num_points = 20
    spline = torch.empty((num_points, 2))
    spline[:, 0], spline[:, 1] = torch.linspace(-0.5, 0.5, num_points), torch.linspace(-0.5, 0.5, num_points)

    def test_energy(gplvm, spline, D=3):
        dmu, dvar = gplvm.derivatives(spline)
        mean_c, var_c = gplvm.embed(spline)
        ncn = NonCentralNakagami(dmu, dvar)

        variance_term = (2 * (var_c.trace() - var_c[1:, 0:-1].trace()) - var_c[0, 0] - var_c[-1, -1]).sum()
        mean_term = (mean_c[1:, :] - mean_c[0:-1, :]).pow(2).sum()

        energy1 = ((dmu.T) @ dmu).trace() + D * (dvar).sum()
        energy2 = mean_term + D * variance_term
        energy3 = (ncn.expectation() ** 2 + ncn.variance()).sum()

        # print(energy1, energy2, energy3)
        torch.testing.assert_allclose(
            energy1, energy2, msg="Not enough data points in spline or error while computing the derivatives"
        )

        torch.testing.assert_allclose(
            energy1, energy3, msg="Error with the derivative or with Non Central Nakagami function"
        )

    # test_energy(gplvm_riemann, spline)

    # J_mu, J_cov = gplvm_riemann.jacobian_posterior(spline)
    # dmu, dvar = gplvm.derivatives(spline)
