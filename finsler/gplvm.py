import numpy as np
import torch
from torch.autograd import functional
from stochman.manifold import Manifold


def print_is_positive_definite(matrix):
    matrix = matrix.detach().numpy()
    print('Is positive definite symmetric: {}'.format(np.all(np.linalg.eigvals(matrix) > 0)))

class gplvm(Manifold):
    """Class that takes a gplvm model as input and allows for computation of curve energy"""

    def __init__(self, object, device=None, mode='riemannian'):
        self.model = object.to(device)   #
        self.device = device
        self.mode = mode        # Riemannian or Finslerian
        self.data = object.base_model.y

    def pairwise_distances(self,x, y=None):
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


    def evaluateKernel(self,xstar,x):
        """
        For the RBF kernel, evaluate the kernel using the pairwise distances
        Returns the evaluated kernel of the latent variables and the new point
        """
        r = self.pairwise_distances(xstar,x)
        var = self.model.base_model.kernel.variance_unconstrained.exp()
        l = self.model.base_model.kernel.lengthscale_unconstrained.exp()

        ksx = var * torch.exp((-0.5*r/l**2))
        return ksx

    def evaluateDiffKernel(self, xstar, X, kernel_type='rbf'):
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
        if kernel_type=='rbf': # faster for RBF
            var = self.model.base_model.kernel.variance_unconstrained.exp()
            l = self.model.base_model.kernel.lengthscale_unconstrained.exp()
            ksx = self.evaluateKernel(xstar,X)
            dK = -l**-2*(xstar-X).T*ksx
            ddK = (l**-2*var)*torch.eye(d).to(torch.float64)
        else:
            # kernel_jac = functional.jacobian(self.evaluateKernel, (xstar, X), create_graph=True)
            # kernel_hes = functional.hessian(self.evaluateKernel, (xstar, xstar), create_graph=True)
            kernel_jac = functional.jacobian(self.model.base_model.kernel.forward, (xstar, X), create_graph=False)
            kernel_hes = functional.hessian(self.model.base_model.kernel.forward, (xstar, xstar), create_graph=False)
            dK = torch.reshape(torch.squeeze(kernel_jac[0]), (N_train, d)).T
            ddK = torch.reshape(torch.squeeze(kernel_hes[0][1]), (d, d))
        return dK, ddK

    def embed_old(self,xstar):
        """ Maps from latent to data space, implements equations 2.23 - 2.24 from Rasmussen.
        We assume a different mean function across dimensions but same covariance matrix
        """
        X = self.model.X_loc
        Y = self.model.base_model.__dict__['y'].clone().detach().t()
        n = X.shape[0]
        noise = self.model.base_model.noise_unconstrained.exp()**2
        X, Y= X.to(torch.float64), Y.to(torch.float64)
        xstar = xstar.to(torch.float64) # Numerical precision needed

        Ksx = self.evaluateKernel(xstar,X)
        Kxx = self.evaluateKernel(X,X) + torch.eye(n)*noise
        Kss = self.evaluateKernel(xstar,xstar)
        Kinv = (Kxx).cholesky().cholesky_inverse() # Kinv = Kxx.inverse()

        mu = Ksx.mm(Kinv).mm(Y)
        Sigma = Kss - Ksx.mm(Kinv).mm(Ksx.T) # should be symm. positive definite
        # print('MU:',mu[0,:].detach().numpy())
        # print('SIGMA:',Sigma[0,0].detach().numpy())
        return mu, Sigma

    def embed_old2(self,xstar,jitter=1e-5):
        """ Maps from latent to data space, implements equations 2.23 - 2.24 from Rasmussen.
        We assume a different mean function across dimensions but same covariance matrix
        """
        X = self.model.X_loc
        Y = self.model.base_model.__dict__['y'].clone().detach().t()
        n = X.shape[0]
        noise = self.model.base_model.noise_unconstrained.exp()**2
        X, Y= X.to(torch.float64), Y.to(torch.float64)
        # HACK: Do we change the type of the tensors?
        xstar = xstar.to(torch.float64) # Numerical precision needed

        Ksx = self.model.base_model.kernel.forward(xstar,X) # (Nstar,N)
        Kxx = self.model.base_model.kernel.forward(X,X) + torch.eye(n).to(self.device)*noise # (N,N)
        Kss = self.model.base_model.kernel.forward(xstar,xstar) # (Nstar,Nstar)
        Kinv = (Kxx).cholesky().cholesky_inverse() # (N,N)

        # Ksx = self.evaluateKernel(xstar,X)
        # Kxx = self.evaluateKernel(X,X) + torch.eye(n)*noise
        # Kss = self.evaluateKernel(xstar,xstar)
        # Kinv = (Kxx).cholesky().cholesky_inverse()

        mu = Ksx.mm(Kinv).mm(Y)
        Sigma = Kss - Ksx.mm(Kinv).mm(Ksx.T) + jitter*torch.eye(xstar.shape[0]) # should be symm. positive definite
        return mu, Sigma

    def embed(self, xstar):
        mu, var = self.model.base_model(xstar.float(),full_cov=False)
        return mu.t(), var[0,:].t()


    def derivatives(self,coords, method='discretization'):
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
        # QUESTION: coords should have dimensions: N_test x d? Yes!
        # QUESTION: isn't the description of the output mixed up? I changed 'sigma' for 'var', to be clearer
        if method == 'discretization':
            mean_c, var_c = self.embed_old2(coords)
            mu_derivatives = mean_c[0:-1,:] - mean_c[1:,:]
            var_derivatives = var_c.diagonal()[1:]+var_c.diagonal()[0:-1] -2*(var_c[0:-1,1:].diagonal())
            # mu, var = mu_{i+1} - mu_{i}, s_{i+1,i+1} + s_{i,i} - 2*s_{i,i+1}

        elif method == 'obs_derivatives':
            dc = coords[0:-1,:] - coords[1:,:] # derivatives of the spline (dc/dt)
            c = (coords[0:-1,:] + coords[1:,:])/2 # coordinates of derivatives
            X = self.model.X_loc
            Y = self.model.base_model.__dict__['y'].clone().detach().t()
            X, Y = X.to(torch.float64), Y.to(torch.float64) # num. precision
            c, dc = c.to(torch.float64), dc.to(torch.float64)
            noise = self.model.base_model.noise_unconstrained.exp()**2
            N, D, d, Ntest = Y.shape[0], Y.shape[1], X.shape[1], dc.shape[0]

            # kxx = self.evaluateKernel(X,X) + torch.eye(N)*noise
            kxx = self.model.base_model.kernel.forward(X,X) + torch.eye(N)*noise
            kinv = (kxx).cholesky().cholesky_inverse()
            mu_derivatives = torch.zeros(Ntest, D)
            var_derivatives = torch.zeros(Ntest)

            for nn in range(Ntest):
                dk, ddk = self.evaluateDiffKernel(c[nn,:].unsqueeze(0), X)
                var_star = ddk - dk.mm(kinv).mm(dk.T) # var(df/dc)
                var = dc[nn,:].unsqueeze(0).mm(var_star).mm(dc[nn,:].unsqueeze(0).T) # var(df/dt) = dc/dt * var(df/dc) * (dc/dt).T
                var_derivatives[nn] = var

                for dd in range(D):
                    y = Y[:,dd].unsqueeze(1)
                    mu_star = dk.mm(kinv).mm(y) # mean(df/dc)
                    mu = dc[nn,:].unsqueeze(0).mm(mu_star) # mean(df/dt) = dc/dt * mean(df/dc)
                    mu_derivatives[nn,dd] = mu
        return mu_derivatives, var_derivatives

    def jacobian_posterior(self, xstar):
        X = self.model.X_loc
        Y = self.model.base_model.__dict__['y'].clone().detach().t()
        X, Y = X.to(torch.float64), Y.to(torch.float64) # num. precision
        xstar = xstar.to(torch.float64)
        # noise = self.model.base_model.noise_unconstrained.exp()**2
        # raise
        noise = 1e-7
        N, D, d, Ntest = Y.shape[0], Y.shape[1], X.shape[1], xstar.shape[0]
        mu_star, var_star = torch.empty(Ntest, d, D), torch.empty(Ntest, d, d)

        # kxx = self.evaluateKernel(X,X) + torch.eye(N)*noise
        kxx = self.model.base_model.kernel.forward(X,X) + torch.eye(N)*noise
        kinv = (kxx).cholesky().cholesky_inverse()
        # print_is_positive_definite(kinv)
        for nn in range(Ntest):
            dk, ddk = self.evaluateDiffKernel(xstar[nn,:].unsqueeze(0), X)
            # print(dk)
            # print(kinv)
            # print(Y)
            # raise
            var_star[nn,:,:] = ddk - dk.mm(kinv).mm(dk.T) # var(df/dc)
            mu_star[nn,:,:] = dk.mm(kinv).mm(Y) # mean(df/dc)
        #     print('---')
        #     print_is_positive_definite(var_star[nn,:,:])
        # raise
        return mu_star.to(torch.float32), var_star.to(torch.float32)


    def curve_energy(self, coords):
        # n = self.model.X_loc.shape[0]
        # D = self.model.base_model.__dict__['y'].shape[0]
        n,D = self.data.shape
        if self.mode == 'riemannian':
            mu, var = self.embed_old2(coords)
            # energy = (mu[1:,:] -mu[0:-1,:]).pow(2).sum() + D*(2*var[1:] - 2*var[0:-1]).sum()
            energy = (mu[1:,:] -mu[0:-1,:]).pow(2).sum() + D*(2*var.trace() - 2*var[1:,0:-1].trace())
        elif self.mode == 'finslerian':
            from .distributions import NonCentralNakagami
            dmu, dvar = self.derivatives(coords)
            non_central_nakagami = NonCentralNakagami(dmu, dvar)
            energy = (non_central_nakagami.expectation()**2).sum()
            # Comparison
            # energy_riemann1 = energy + (non_central_nakagami.variance()).sum()
            # mu, var = self.embed(coords)
            # energy_riemann2 = (dmu@dmu.t()).trace()+D*(dvar**2).sum()
            # print('{} energy_riemann1: {:.4f}'.format(self.mode, energy_riemann1.detach().numpy()))
            # print('{} energy_riemann2: {:.4f}'.format(self.mode, energy_riemann2.detach().numpy()))
        print('{} energy: {:.4f}'.format(self.mode, energy.detach().numpy()))
        # print('{} energy: {:.4f} \r'.format(self.mode, energy.detach().numpy()), end='\r')
        return energy


    def evaluateDiffKernel_batch(self, xstar, X, kernel_type='rbf'):
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
        if kernel_type=='rbf': # faster for RBF
            if len(xstar.shape) <2:
                xstar = xstar.unsqueeze(0)
            nof_points =  xstar.shape[0]

            var = self.model.base_model.kernel.variance_unconstrained.exp()
            l = self.model.base_model.kernel.lengthscale_unconstrained.exp()

            ksx = self.model.base_model.kernel.forward(xstar,X)

            X = torch.cat(nof_points*[X.unsqueeze(0)])
            xstar = torch.cat(N_train*[xstar.unsqueeze(0)]).transpose(0,1)

            dK_term1 = xstar-X
            dK_term2 = torch.cat(D*[ksx.unsqueeze(0)]).transpose(0,1).transpose(1,2)

            dK = -l**-2*dK_term1*dK_term2
            ddK = (l**-2*var)*torch.eye(D).to(self.device)
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

    def jacobian(self,xstar):
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

        sigma2 = self.model.base_model.noise_unconstrained.exp()**2
        if sigma2<1e-4: sigma2=1e-4
        N, d = x.shape
        Y = self.data#.t()
        D = Y.shape[0]
        x, Y = x.to(torch.float64), Y.to(torch.float64) # num. precision
        xstar = xstar.to(torch.float64)

        nof_points =  xstar.shape[0]

        dk, ddk = self.evaluateDiffKernel_batch(xstar, x)
        kxx = self.model.base_model.kernel.forward(x,x) + torch.eye(N).to(self.device)*sigma2

        kinv = (kxx).cholesky().cholesky_inverse()
        ksx = self.model.base_model.kernel.forward(xstar,x)

        mu_star = torch.matmul(dk.transpose(1,2),kinv.mm(Y.t()))
        cov_star = ddk - torch.bmm(torch.matmul(dk.transpose(1,2),kinv),dk)
        # if nof_points == 1:
        #     mu_star = mu_star.squeeze()
        #     cov_star = cov_star.squeeze()
        return mu_star, cov_star

    def sample_jacobian(self,xstar,nof_samples=1):
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
            return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0), A.size(1)*B.size(1))

        d = self.model.X_loc.shape[1]
        Y = self.model.base_model.__dict__['y'].clone().detach().t()
        D = Y.shape[1]

        if len(xstar.shape) <2:
            xstar = xstar.unsqueeze(0)
        nof_points =  xstar.shape[0]

        # mu_J,cov_J = self.jacobian(xstar)
        mu_J,cov_J = self.jacobian_posterior(xstar)

        j_samples_by_point = list() # list of samples of each point, length: nof_samples
        for sample_index in range(nof_samples):
            j_by_point = list() # list of j by point, length: nof_points
            for point_index in range(nof_points):
                mean = mu_J[point_index,:].view(1,-1).squeeze()
                cov = kronecker(cov_J[point_index,:],torch.eye(D))
                cov  = torch.cholesky(cov) + torch.cholesky(cov.t(),upper=True) # force symmetry

                # Haven't tested that the mean should be explicitly be converted to double but I (cife) would think this is correct
                J = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
                #J = torch.distributions.multivariate_normal.MultivariateNormal(mean.double(), covariance_matrix=cov.double())
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

    def metric(self,xstar):
        """Computes the **expected** Riemannian metric at points xstar as the metric has to be deterministic
        - input: Points at which to compute the expected metric. Should be of size nof_points x d
        - output: Expected metric tensor of shape nof_points x d x d
        Ref: Tosi (2014), eq 22: https://arxiv.org/abs/1411.7432
        """
        J_mu, J_cov = self.jacobian_posterior(xstar)
        JJ_mu = torch.bmm(J_mu,J_mu.transpose(1,2))
        D = self.data.shape[0]
        # J_cov = 0
        # G_expected = self.data.shape[0]*cov_J + torch.bmm(mu_Js,mu_Js.transpose(1,2))
        # G_expected = (1./self.data.shape[1])*torch.bmm(mu_Js,mu_Js.transpose(1,2)) + cov_J
        return D*J_cov + JJ_mu

    def sample_metric(self,xstar,nof_samples=50):
        """Allows sampling from the Riemannian metric at multiple points.
        - input:
            xstar: Points at which to compute the expected metric. Should be of size nof_points x d
            nof_samples: Number of desired samples
        - output:
            g_sample: Samples of the metric. Size (nof_points, nof_samples, d x d)
        """
        j_samples = self.sample_jacobian(xstar,nof_samples=nof_samples)
        g_sample = torch.matmul(j_samples,j_samples.transpose(2,3))
        # return g_sample
        return torch.mean(g_sample, dim=0)
