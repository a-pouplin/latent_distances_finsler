# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import numpy as np
import torch
from torch.distributions.normal import Normal

from finsler.likelihoods.likelihood import Likelihood


class Gaussian(Likelihood):
    """
    Class for Gaussian Likelihood
    """

    def __init__(self, sigma=None, fit_noise=False, dim=None):
        super(Gaussian, self).__init__()

        if sigma is None:
            log_sigma = torch.log(torch.tensor(1.0))
        else:
            log_sigma = torch.log(torch.tensor(sigma))

        # Dimensionality of likelihood observation model
        if dim is None:
            self.dim = 1

        self.log_sigma = torch.nn.Parameter(log_sigma * torch.ones(1), requires_grad=fit_noise)

    def pdf(self, f, y):
        sigma = torch.exp(self.sigma).abs().clamp(min=0.1, max=100.0)  # min-max enforced
        normal = Normal(loc=f, scale=sigma)
        pdf = torch.exp(normal.log_prob(y))
        return pdf

    def logpdf(self, f, y):
        sigma = torch.exp(self.sigma).abs().clamp(min=0.1, max=100.0)  # min-max enforced
        normal = Normal(loc=f, scale=sigma)
        logpdf = normal.log_prob(y)
        return logpdf

    def variational_expectation(self, y, m, v):
        # Variational Expectation of log-likelihood -- Analytical
        sigma = torch.exp(self.sigma).abs().clamp(min=0.1, max=100.0)  # min-max enforced
        lik_variance = sigma.pow(2)
        expectation = (
            -np.log(2 * np.pi) - torch.log(lik_variance) - (y.pow(2) + m.pow(2) + v - (2 * m * y)).div(lik_variance)
        )

        return 0.5 * expectation

    def log_predictive(self, y_test, mu_gp, v_gp, num_samples=1000):
        # function samples:
        normal = Normal(loc=mu_gp.flatten(), scale=torch.sqrt(v_gp).flatten())
        f_samples = normal.sample(sample_shape=(1, num_samples))[0, :, :]

        # monte-carlo:
        logpdf = self.logpdf(f_samples, y_test.flatten())
        log_pred = -np.log(num_samples) + torch.logsumexp(logpdf, dim=0)
        return log_pred


class MultivariateGaussian(Likelihood):
    """
    Class for Multivariate Gaussian Likelihood
    """

    def __init__(self, sigma=None, fit_noise=False, dim=None):
        super(MultivariateGaussian, self).__init__()

        if dim is None:
            self.dim = 2
        else:
            self.dim = dim

        if sigma is None:
            sigma = 1.0

        self.sigma = torch.nn.Parameter(sigma * torch.ones(1), requires_grad=fit_noise)

    def pdf(self, f, y):
        N, D = y.shape
        y_vector = y.view(N * D, 1)
        f_vector = f.view(N * D, 1)
        normal = Normal(loc=f_vector, scale=self.sigma)
        pdf = torch.exp(normal.log_prob(y_vector))
        return pdf.view(N, D)

    def logpdf(self, f, y):
        N, D = y.shape
        y_vector = y.view(N * D, 1)
        f_vector = f.view(N * D, 1)
        normal = Normal(loc=f_vector, scale=self.sigma)
        logpdf = normal.log_prob(y_vector)
        return logpdf.view(N, D)

    def variational_expectation(self, y, m, v):
        # Variational Expectation of log-likelihood -- Analytical
        N, D = y.shape
        y_vector = y.view(N * D, 1)
        m_vector = m.view(N * D, 1)
        v_vector = v.view(N * D, 1)
        lik_variance = self.sigma.pow(2)
        expectation = (
            -np.log(2 * np.pi)
            - torch.log(lik_variance)
            - (y_vector.pow(2) + m_vector.pow(2) + v_vector - (2 * m_vector * y_vector)).div(lik_variance)
        )

        return 0.5 * expectation.view(N, D)
