# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pablo Moreno-Munoz, Cilie W. Feldager
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import MultivariateNormal as Normal
from torch.utils.data import DataLoader


class SASGP(nn.Module):
    """
    -- * SASGP: Stochastic Active Set GP [Latent Variable Model]

        Description:    Main Class
        ----------
        Parameters
        ----------
        - kernel:       object of kernel class (see run.py)
        - likelihood:   object of likelihood class (see run.py)
        - data_dim:     int / dimensionality of data X
        - latent_dim:   int / dimensionality of latent space Z
        - data_size:    int / size of dataset (for SGD)
        - learning_rate:float / for Adam optimizer {1e-2,1e-3} recommended
        - active_set:   int / size of the active set A
        - device:       string / {'cpu', 'gpu'}
    """

    def __init__(
        self,
        kernel,
        likelihood,
        data_dim=None,
        latent_dim=None,
        data_size=None,
        learning_rate=1e-2,
        active_set=20,
        device="cpu",
    ):
        super(SASGP, self).__init__()

        # Dimensions
        if latent_dim is None:
            latent_dim = 2

        self.latent_dim = int(latent_dim)  # dimensionality of latent space z
        self.active_set = active_set

        if data_size is None:
            raise AssertionError
        else:
            self.data_size = data_size

        if data_dim is None:
            self.data_dim = 784
        else:
            self.data_dim = data_dim

        # Gaussian Process (GP) Core Elements
        self.likelihood = likelihood
        self.kernel = kernel

        # Amortization Net
        self.amortization_net = nn.Sequential(
            nn.Linear(self.data_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, self.latent_dim)
        )

        # Optimization setup
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if device == "gpu":
            self.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    # Adapt the following ones to
    def active_set_permutation(self, x):
        """Description:    Does a random permutation of data and selects a subset
        Input:          Data observations X (NxD)
        Return:         Active Set X_A and X_rest / X_A U X_rest = X
        """
        permutation = torch.randperm(x.size()[0])
        indices = permutation[: self.active_set]
        rest = permutation[self.active_set :]
        a = x[indices]
        x_rest = x[rest]
        return a, x_rest

    # def embed(self, xstar):
    #     """
    #     Returns the mean and covariance functions evaluated on points in the latent space.
    #     Input:
    #         xstar:      Points (Nstar) in the latent space where the GP should be evaluated
    #                     Shape: (Nstar,q)
    #         jitter:     A small number added to the diagonal to ensure that the covariance matrix is psd.
    #     Output:
    #         mu:         Mean (Nstar,D)
    #         Sigma:      Covariance (Nstar,Nstar)
    #     Maps from latent to data space, implements equations 2.23 - 2.24 from Rasmussen.
    #     We assume a different mean function across dimensions but same covariance matrix
    #     """
    #     Kss = self.kernel.K(xstar)
    #     Kxx = self.kernel.K(self.x)
    #     # iKxx, _ = torch.solve(torch.eye(self.N), Kxx)
    #     iKxx = torch.linalg.solve(torch.eye(self.N), Kxx)
    #     Ksx = self.kernel.K(xstar, self.x)

    #     mu = Ksx.mm(iKxx).mm(self.y)
    #     Sigma = Kss - Ksx.mm(iKxx).mm(Ksx.t())

    #     mu_vector = mu
    #     Sigma_matrix = Sigma

    #     return mu_vector, Sigma_matrix

    def forward(self, x):
        """Description:    Forward pass of the model (Pytorch)
        Input:          Data observations X (NxD)
        Return:         Loss function (Log-marginal likelihood) for all cases of A
        """
        # x = batch of data, a = active set
        a, x_rest = self.active_set_permutation(x)
        batch_size = x.shape[0]
        rest_size = x_rest.shape[0]
        sgd_constant = rest_size / self.data_size

        if x_rest.shape[0] > 0:

            # Latent Variable Amortization
            x_fold = x_rest.view(x_rest.shape[0], -1)
            a_fold = a.view(a.shape[0], -1)

            z = self.amortization_net(x_fold)
            za = self.amortization_net(a_fold)

            # Log-likelihood computation for active set
            Kaa = self.kernel.K(za, za) + torch.exp(self.likelihood.log_sigma) * torch.eye(self.active_set)

            aaT = a.matmul(a.t())
            La = self.kernel.jitchol(Kaa)
            A = torch.cholesky_solve(aaT, La)
            trA = torch.trace(A)

            log_likelihood = -0.5 * self.latent_dim * self.active_set
            log_likelihood += -0.5 * self.latent_dim * torch.logdet(Kaa) - 0.5 * trA

            # Rest
            Knn = self.kernel.K(z, z) + torch.exp(self.likelihood.log_sigma) * torch.eye(z.shape[0])
            Kna = self.kernel.K(z, za)

            Q = torch.cholesky_solve(Kna.t(), La)
            m_a = Q.t().matmul(a)
            c_a = torch.diag(Knn) - (Q.t() * Kna).sum(1)
            var_a = c_a[:, None].repeat(1, x_rest.shape[1])  # augmentation to be N x D

            log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * torch.log(var_a) - 0.5 * (x_rest - m_a**2).div(var_a)
            log_likelihood += sgd_constant * log_prob.sum()

        else:
            # Latent Variable Amortization
            a_fold = a.view(a.shape[0], -1)

            za = self.amortization_net(a_fold)

            # Log-likelihood computation for active set
            Kaa = self.kernel.K(za, za) + torch.exp(self.likelihood.log_sigma) * torch.eye(za.shape[0])

            aaT = a.matmul(a.t())
            La = self.kernel.jitchol(Kaa)
            A = torch.cholesky_solve(aaT, La)
            trA = torch.trace(A)

            log_likelihood = -0.5 * self.latent_dim * self.active_set
            log_likelihood += -0.5 * self.latent_dim * torch.logdet(Kaa) - 0.5 * trA

        return -log_likelihood

    def exact_log_likelihood(self, x):
        """Description:    Exact log-marginal likelihood of the model (from original GPLVM model)
                        Attention -- This inverts a covariance matrix NxN
        Input:          Data observations X (NxD)
        Return:         Exact marginal log-likelihood p(x)
        """
        x_fold = x.view(x.shape[0], -1)
        z = self.amortization_net(x_fold)

        # Log-likelihood computation for active set
        Knn = self.kernel.K(z, z) + self.likelihood.sigma * torch.eye(self.data_size)

        xxT = x.matmul(x.t())
        # L = torch.linalg.cholesky(Knn)
        L = self.kernel.jitchol(Knn)
        Q = torch.cholesky_solve(xxT, L)
        trQ = torch.trace(Q)

        log_likelihood = -0.5 * self.latent_dim * self.active_set
        log_likelihood += -0.5 * self.latent_dim * torch.logdet(Knn) - 0.5 * trQ

        return -log_likelihood

    def predictive(self, x_star, x):
        """Description:    Exact Posterior predictive computation of the GPLVM model
                        Attention -- This inverts a covariance matrix NxN
        Input:          x:      Data observations X (N x D)
                        x_star: Data to predict over X_star (N_star x D)
        Return:         Exact marginal log-likelihood logp(x)
        """
        x_fold = x.view(x.shape[0], -1)
        z = self.amortization_net(x_fold)

        x_star_fold = x_star.view(x_star.shape[0], -1)
        z_star = self.amortization_net(x_star_fold)

        Kss = self.kernel.K(z_star, z_star) + self.likelihood.sigma * torch.eye(z_star.shape[0])
        Knn = self.kernel.K(z, z) + self.likelihood.sigma * torch.eye(z.shape[0])
        Kns = self.kernel.K(z, z_star)

        L = self.kernel.jitchol(Knn)
        Q = torch.cholesky_solve(Kns, L)

        m_star = Q.t().matmul(x)
        c_star = torch.diag(Kss) - (Q.t() * Kns.t()).sum(1)
        var_star = c_star[:, None].repeat(1, self.data_dim)  # augmentation to be N x D

        return m_star, var_star

    def rmse(self, x_star, x):
        """Description:    Root Mean Square Error (rmse)
        Input:          x:      Data observations X (N x D)
                        x_star: Data to predict over X_star (N_star x D)
        Return:         rmse
        """
        x_fold = x_star.view(x_star.shape[0], -1)
        mu_star, _ = self.predictive(x_star, x)
        rmse = torch.sqrt(torch.mean((x_fold - mu_star) ** 2))
        return rmse

    def mae(self, x_star, x):
        """Description:    Mean Absolute Error (mae)
        Input:          x:      Data observations X (N x D)
                        x_star: Data to predict over X_star (N_star x D)
        Return:         mae
        """
        x_fold = x_star.view(x_star.shape[0], -1)
        mu_star, _ = self.predictive(x_star, x)
        mae = torch.mean(torch.abs(x_fold - mu_star))
        return mae

    def nlpd(self, x_star, x):
        """Description:    Negative Log-Predictive Density (nlpd)
        Input:          x:      Data observations X (N x D)
                        x_star: Data to predict over X_star (N_star x D)
        Return:         nlpd
        """
        x_fold = x_star.view(x_star.shape[0], -1)
        mu_star, var_star = self.predictive(x_star, x)
        nlpd = torch.abs(
            0.5 * math.log(2 * math.pi) + 0.5 * torch.mean(torch.log(var_star) + (x_fold - mu_star) ** 2 / var_star)
        )
        return nlpd

    def error_metrics(self, x_star, x):
        """Description:    Computes all metrics at once {rmse, mae, nlpd}
        Input:          x:      Data observations X (N x D)
                        x_star: Data to predict over X_star (N_star x D)
        Return:         rmse, mae, nlpd
        """
        rmse = self.rmse(x_star, x)
        mae = self.mae(x_star, x)
        nlpd = self.nlpd(x_star, x)
        return rmse, mae, nlpd


class BayesianSASGP(nn.Module):
    """
    -- * BayesianSASGP: Stochastic Active Set Bayesian GP [Latent Variable Model]

        Description:    Main Class
        ----------
        Parameters
        ----------
        - kernel:               object of kernel class (see run.py)
        - likelihood:           object of likelihood class (see run.py)
        - data_dim:             int / dimensionality of data X
        - latent_dim:           int / dimensionality of latent space Z
        - data_size:            int / size of dataset (for SGD)
        - learning_rate:        float / for Adam optimizer {1e-2,1e-3} recommended
        - active_set:           int / size of the active set A
        - device:               string / {'cpu', 'gpu'}
        - small_amortization:   bool / make the amortization net <very> small?
    """

    def __init__(
        self,
        kernel,
        likelihood,
        data_dim=None,
        latent_dim=None,
        data_size=None,
        learning_rate=1e-2,
        active_set=20,
        device="cpu",
        small_amortization=True,
    ):
        super(BayesianSASGP, self).__init__()

        # Dimensions
        if latent_dim is None:
            latent_dim = 2

        self.latent_dim = int(latent_dim)  # dimensionality of latent space z
        self.active_set = active_set

        if data_size is None:
            raise AssertionError
        else:
            self.data_size = data_size

        if data_dim is None:
            self.data_dim = 784
        else:
            self.data_dim = data_dim

        # Gaussian Process (GP) Core Elements
        self.likelihood = likelihood
        self.kernel = kernel

        # Amortized Variational Distribution q(z|x)
        if small_amortization:
            self.mu_z = nn.Sequential(nn.Linear(self.data_dim, 50), nn.Tanh(), nn.Linear(50, self.latent_dim))

            self.var_z = nn.Sequential(
                nn.Linear(self.data_dim, 50), nn.Tanh(), nn.Linear(50, self.latent_dim), nn.Softplus()
            )
        else:
            self.mu_z = nn.Sequential(
                nn.Linear(self.data_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.latent_dim),
            )

            self.var_z = nn.Sequential(
                nn.Linear(self.data_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.latent_dim),
                nn.Softplus(),
            )

        # Optimization setup
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if device == "gpu":
            self.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def divergence_kl(self, mu_z, var_z):
        """Description:    KL divergence between prior and variational posterior
        Input:          mu_z:   Posterior mean      (N x Latent Dim)
                        var_z:  Posterior variance  (N x Latent Dim)
        Return:         Kullback-Leibler Divergence
        """
        kl_z = -0.5 * torch.sum(1 + var_z.log() - mu_z.pow(2) - var_z)
        return kl_z

    def active_set_permutation(self, x):
        """Description:    Does a random permutation of data and selects a subset
        Input:          Data observations X (NxD)
        Return:         Active Set X_A and X_rest / X_A U X_rest = X
        """
        permutation = torch.randperm(x.size()[0])

        indices = permutation[: self.active_set]
        rest = permutation[self.active_set :]
        a = x[indices]
        x_rest = x[rest]

        return a, x_rest

    def expectation(self, z_s, za_s, x_rest, a):
        rest_size = x_rest.shape[0]
        sgd_constant = rest_size / self.data_size

        Kaa = self.kernel.K(za_s, za_s) + torch.exp(self.likelihood.log_sigma) * torch.eye(self.active_set)
        La = self.kernel.jitchol(Kaa)

        Knn = self.kernel.K(z_s, z_s) + torch.exp(self.likelihood.log_sigma) * torch.eye(rest_size)
        Kna = self.kernel.K(z_s, za_s)

        Q = torch.cholesky_solve(Kna.t(), La)
        m_a = Q.t().matmul(a)
        c_a = torch.diag(Knn) - (Q.t() * Kna).sum(1)
        var_a = c_a[:, None].repeat(1, x_rest.shape[1])  # augmentation to be N x D

        log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * torch.log(var_a) - 0.5 * (x_rest - m_a**2).div(var_a)

        exp_conditional = sgd_constant * log_prob.sum()

        active_dist = Normal(torch.zeros(self.active_set), Kaa)
        exp_active_set = active_dist.log_prob(a.t()).sum()

        expectation = exp_active_set + exp_conditional

        return expectation

    def forward(self, x):
        """Description:    Forward pass of the model (Pytorch)
        Input:          Data observations X (NxD)
        Return:         Loss function (ELBO) for all cases of A
        """
        # x: batch of data
        a, x_rest = self.active_set_permutation(x)

        if x_rest.shape[0] > 0:
            rest_size = x_rest.shape[0]
            sgd_constant = rest_size / self.data_size

            # Variational Distribution Amortization
            x_fold = x_rest.view(x_rest.shape[0], -1)
            a_fold = a.view(a.shape[0], -1)

            # build q(z) distribution for x_batch
            mu_z = self.mu_z(x_fold)
            var_z = self.var_z(x_fold)

            # build q(z) distribution for active_set // this computation could be saved
            mu_za = self.mu_z(a_fold)
            var_za = self.var_z(a_fold)

            # Expectation term
            z_sample = mu_z + torch.randn_like(var_z) * var_z.sqrt()
            za_sample = mu_za + torch.randn_like(var_za) * var_za.sqrt()  # this computation could be saved
            expectation = self.expectation(z_sample, za_sample, x_rest, a)

            # KL Divergence w.r.t. x and a
            kl_z = sgd_constant * self.divergence_kl(mu_z, var_z)
            kl_a = self.divergence_kl(mu_za, var_za)  # this computation could be saved

            elbo = expectation - kl_z - kl_a

        else:
            # Variational Distribution Amortization
            a_fold = a.view(a.shape[0], -1)

            # build q(z) distribution for active_set // this computation could be saved
            mu_za = self.mu_z(a_fold)
            var_za = self.var_z(a_fold)

            # Expectation term
            za_sample = mu_za + torch.randn_like(var_za) * var_za.sqrt()  # this computation could be saved
            Kaa = self.kernel.K(za_sample, za_sample) + torch.exp(self.likelihood.log_sigma) * torch.eye(
                za_sample.shape[0]
            )

            active_dist = Normal(torch.zeros(za_sample.shape[0]), Kaa)
            exp_active_set = active_dist.log_prob(a.t()).sum()
            expectation = exp_active_set

            # KL Divergence w.r.t. x and a
            kl_a = self.divergence_kl(mu_za, var_za)  # this computation could be saved
            elbo = expectation - kl_a

        return -elbo

    def predictive(self, x_star, x):
        """Description:    Exact Posterior predictive computation of the GPLVM model
                        Attention -- This inverts a covariance matrix NxN
        Input:          x:      Data observations X (N x D)
                        x_star: Data to predict over X_star (N_star x D)
        Return:         rmse
        """
        x_fold = x.view(x.shape[0], -1)
        mu_z = self.mu_z(x_fold)
        var_z = self.var_z(x_fold)

        x_star_fold = x_star.view(x_star.shape[0], -1)
        mu_z_star = self.mu_z(x_star_fold)
        var_z_star = self.var_z(x_star_fold)

        z = mu_z + torch.randn_like(var_z) * var_z.sqrt()  # this computation could be saved
        z_star = mu_z_star + torch.randn_like(var_z_star) * var_z_star.sqrt()  # this computation could be saved

        Knn = self.kernel.K(z, z) + self.likelihood.sigma * torch.eye(z.shape[0])
        Kss = self.kernel.K(z_star, z_star) + self.likelihood.sigma * torch.eye(z_star.shape[0])
        Kns = self.kernel.K(z, z_star)

        L = self.kernel.jitchol(Knn)
        Q = torch.cholesky_solve(Kns, L)

        m_star = Q.t().matmul(x)
        c_star = torch.diag(Kss) - (Q.t() * Kns.t()).sum(1)
        var_star = c_star[:, None]  # .repeat(1, self.data_dim)  # augmentation to be N x D

        return m_star, var_star

    def predictive_data_precision(self, z_star, x):
        """Description:    Plots the precision indicated by p(f|x) for
                        for colored-surface plots of uncertainty
                        on top of latent space
        Input:          x:     Data observations X (N x D)
                        z_star: Latent coordinate to predict on (N x D)
        Return:         Precision of the predictive posterior
        """
        data_loader = DataLoader(x, batch_size=self.active_set, pin_memory=True, shuffle=True)
        pred_precision = torch.zeros(z_star.shape[0])
        k = 0
        for k, a in enumerate(data_loader):
            a = a[0].squeeze().view(-1, self.data_dim)  # N_batch x dim
            a = Variable(a).float()

            a_fold = a.view(a.shape[0], -1)

            # build q(z) distribution for a
            mu_za = self.mu_z(a_fold)
            var_za = self.var_z(a_fold)

            # Expectation term
            za_sample = mu_za + torch.randn_like(var_za) * var_za.sqrt()  # this computation could be saved
            Kaa = self.kernel.K(za_sample, za_sample) + self.likelihood.sigma * torch.eye(za_sample.shape[0])
            La = self.kernel.jitchol(Kaa)

            Kss = self.kernel.K(z_star, z_star)
            Ksa = self.kernel.K(z_star, za_sample)

            Q = torch.cholesky_solve(Ksa.t(), La)
            c_s = torch.diag(Kss) - (Q.t() * Ksa).sum(1)

            pred_precision += 1 / c_s

        pred_precision = pred_precision / (k + 1)
        return pred_precision

    def rmse(self, x_star, x):
        """Description:    Root Mean Square Error (rmse)
        Input:          x:      Data observations X (N x D)
                        x_star: Data to predict over X_star (N_star x D)
        Return:         rmse
        """
        x_fold = x_star.view(x_star.shape[0], -1)
        mu_star, _ = self.predictive(x_star, x)
        rmse = torch.sqrt(torch.mean((x_fold - mu_star) ** 2))
        return rmse

    def mae(self, x_star, x):
        """Description:    Mean Absolute Error (mae)
        Input:          x:      Data observations X (N x D)
                        x_star: Data to predict over X_star (N_star x D)
        Return:         mae
        """
        x_fold = x_star.view(x_star.shape[0], -1)
        mu_star, _ = self.predictive(x_star, x)
        mae = torch.mean(torch.abs(x_fold - mu_star))
        return mae

    def nlpd(self, x_star, x):
        """Description:    Negative Log-Predictive Density (nlpd)
        Input:          x:      Data observations X (N x D)
                        x_star: Data to predict over X_star (N_star x D)
        Return:         nlpd
        """
        x_fold = x_star.view(x_star.shape[0], -1)
        mu_star, var_star = self.predictive(x_star, x)
        nlpd = torch.abs(
            0.5 * math.log(2 * math.pi) + 0.5 * torch.mean(torch.log(var_star) + (x_fold - mu_star) ** 2 / var_star)
        )
        return nlpd

    def error_metrics(self, x_star, x):
        """Description:    Computes all metrics at once {rmse, mae, nlpd}
        Input:          x:      Data observations X (N x D)
                        x_star: Data to predict over X_star (N_star x D)
        Return:         rmse, mae, nlpd
        """
        rmse = self.rmse(x_star, x)
        mae = self.mae(x_star, x)
        nlpd = self.nlpd(x_star, x)
        return rmse, mae, nlpd


class NonAmortizedSASGP:
    pass


class NonAmortizedBayesianSASGP:
    pass
