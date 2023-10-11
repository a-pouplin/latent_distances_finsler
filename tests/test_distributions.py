from unittest import TestCase

import numpy as np
import torch
from scipy.special import gamma, hyp1f1
from torch.distributions import Normal

from finsler.distributions import NonCentralNakagami

# class function to test dostributions functions


class Test_distributions(TestCase):
    def setUp(self):
        # create a random tensor z ~ N(mu, var)
        self.mu = torch.randn(10, 2)
        self.var = torch.abs(torch.randn(10))
        self.dimensions = self.mu.shape[1]
        self.gamma_ratio = gamma((self.dimensions + 1) / 2) / gamma(self.dimensions / 2)
        self.omega = (self.mu**2).sum(1) / self.var
        self.hyp1f1 = hyp1f1(-1 / 2, self.dimensions / 2, -1 / 2 * self.omega)
        self.ncn = NonCentralNakagami(self.mu, self.var)

    def test_variables_type(self):
        # check if the variables are of the right type
        self.assertIsInstance(self.mu, torch.Tensor)
        self.assertIsInstance(self.var, torch.Tensor)
        self.assertIsInstance(self.dimensions, int)
        self.assertIsInstance(self.gamma_ratio, float)
        self.assertIsInstance(self.omega, torch.Tensor)
        self.assertIsInstance(self.hyp1f1, torch.Tensor)
        self.assertIsInstance(self.ncn, NonCentralNakagami)

    def test_variables_shape(self):
        # check if the variables are of the right shape
        self.assertEqual(self.mu.shape, (10, 2))
        self.assertEqual(self.var.shape, (10,))
        self.assertEqual(self.omega.shape, (10,))
        self.assertEqual(self.hyp1f1.shape, (10,))

    def test_expectation(self):
        # expectation = E[|z|], when z ~ N(mu, var)
        # compute the expectation using the formula
        expectation = np.sqrt(2) * torch.sqrt(self.var) * self.gamma_ratio * self.hyp1f1
        torch.testing.assert_allclose(
            expectation, self.ncn.expectation(), msg="Error in the expectation of the NonCentralNakagami distribution"
        )

    def test_variance(self):
        # variance = var[|z|], when z ~ N(mu, var)
        # compute the variance using the formula
        variance = self.var * (self.omega + self.dimensions - 2 * (self.gamma_ratio * self.hyp1f1) ** 2)
        torch.testing.assert_allclose(
            variance, self.ncn.variance(), msg="Error in the variance of the NonCentralNakagami distribution"
        )

    def test_H1f1Gradients(self):
        pass

    def test_NonCentralNakagami(self):
        D = 3
        nsamples = 10000
        mus = torch.stack([torch.rand(1) for i in range(D)], axis=1)  # list of means
        var = torch.abs(torch.rand(1))  # isotropic variance
        zs = [Normal(loc=mus[:, i], scale=var) for i in range(D)]  # list of normal distributions
        zs_sample = torch.stack([torch.squeeze(z.sample((nsamples,))) for z in zs])  # list of samples

        torch.testing.assert_close(
            torch.mean(zs_sample, axis=1),
            torch.squeeze(mus),
            atol=0.0,
            rtol=0.1,
            msg="Error in the mean of the Normal distribution",
        )

        xs = torch.stack([torch.squeeze(z.sample((nsamples,))) ** 2 for z in zs])
        x = torch.sum(xs, axis=0)  # Wishart W_1 distribution with nsamples
        y = torch.sqrt(x)  # Nakagami distribution with nsamples

        ncnakagami = NonCentralNakagami(mus, var)
        torch.testing.assert_close(
            torch.mean(y),
            ncnakagami.expectation().squeeze(),
            atol=0.0,
            rtol=0.2,
            msg="Error in the expectation of the NonCentralNakagami distribution",
        )

        torch.testing.assert_close(
            torch.var(y),
            ncnakagami.variance().squeeze(),
            atol=0.0,
            rtol=0.2,
            msg="Error in the variance of the NonCentralNakagami distribution",
        )
