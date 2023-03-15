from unittest import TestCase

import numpy as np
import torch

from finsler.distributions import NonCentralNakagami
from finsler.gplvm import Gplvm
from finsler.utils.helper import pickle_load, to_np


class Test_derivatives(TestCase):
    def setUp(self):
        # load previously training gplvm model
        model_saved = pickle_load(folder_path="models/qPCR/", file_name="model.pkl")
        model = model_saved["model"]
        model = model.base_model

        # get Riemannian and Finslerian manifolds
        self.gplvm = Gplvm(model, mode="riemannian")
        num_points = 50
        spline = torch.empty((num_points, 2))
        spline[:, 0], spline[:, 1] = torch.linspace(-0.5, 0.5, num_points), torch.linspace(-0.5, 0.5, num_points)
        self.spline = spline

    def test_derivatives(self):
        pass

    #     """Test the derivatives of the Riemannian manifold"""
    #     deriv_discrete = self.gplvm.derivatives(self.spline, method="discretization")
    #     deriv_diffgp = self.gplvm.derivatives(self.spline, method="obs_derivatives")

    #     torch.testing.assert_close(
    #         deriv_discrete, deriv_diffgp,
    #         check_dtype=False, atol=0., rtol=0.2,
    #         msg="Not enough data points in spline or error while computing the derivatives"
    #     )


class Test_energy(TestCase):
    def setUp(self):
        # load previously training gplvm model
        model = pickle_load(folder_path="models/starfish/", file_name="model.pkl")

        # get Riemannian and Finslerian manifolds
        self.gplvm_riemann = Gplvm(model, mode="riemannian")
        self.gplvm_finsler = Gplvm(model, mode="finslerian")

        num_points = 50
        spline = torch.empty((num_points, 2))
        spline[:, 0], spline[:, 1] = torch.linspace(0, 1, num_points), torch.linspace(0, 1, num_points)
        self.spline = spline

    def compute_energy(self, gplvm, spline, D=3):
        """
        Compute the energy of the manifold
        There are two or three ways to compute the energy, depending on the manifold.
        """
        dmu, dvar = gplvm.derivatives(spline)
        mean_c, var_c = gplvm.embed(spline)
        ncn = NonCentralNakagami(dmu, dvar)

        variance_term = (2 * (var_c.trace() - var_c[1:, 0:-1].trace()) - var_c[0, 0] - var_c[-1, -1]).sum()
        mean_term = (mean_c[1:, :] - mean_c[0:-1, :]).pow(2).sum()

        energy1 = ((dmu.T) @ dmu).trace() + D * (dvar).sum()
        energy2 = mean_term + D * variance_term
        energy3 = (torch.nan_to_num(ncn.expectation(), nan=0) ** 2 + torch.nan_to_num(ncn.variance(), posinf=0)).sum()
        energy1, energy2, energy3 = energy1.detach(), energy2.detach(), energy3.detach()
        return energy1, energy2, energy3

    def test_energy_riemann(self):
        """Test the energy of the Riemannian manifold"""
        energy1, energy2, _ = self.compute_energy(self.gplvm_riemann, self.spline)
        torch.testing.assert_allclose(
            energy1, energy2, msg="Not enough data points in spline or error while computing the derivatives"
        )

    def test_energy_finsler(self):
        """Test the energy of the Finslerian manifold"""
        energy1, energy2, energy3 = self.compute_energy(self.gplvm_finsler, self.spline)

        torch.testing.assert_close(
            energy1, energy2, msg="Not enough data points in spline or error while computing the derivatives"
        )

        torch.testing.assert_close(
            to_np(energy1),
            to_np(energy3),
            msg="Error with the derivative or with Non Central Nakagami function",
            rtol=0.0,
            atol=0.1,
        )
