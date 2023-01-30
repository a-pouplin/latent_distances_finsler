from unittest import TestCase

import torch

from finsler.distributions import NonCentralNakagami
from finsler.gplvm import Gplvm
from finsler.utils.helper import pickle_load


class Test_energy(TestCase):
    def setup(self):
        # load previously training gplvm model
        model_saved = pickle_load(folder_path="trained_models/starfish/", file_name="model.pkl")
        model = model_saved["model"]
        model = model.base_model

        # get Riemannian and Finslerian manifolds
        self.gplvm_riemann = Gplvm(model, mode="riemannian")
        self.gplvm_finsler = Gplvm(model, mode="finslerian")

        num_points = 20
        spline = torch.empty((num_points, 2))
        spline[:, 0], spline[:, 1] = torch.linspace(-0.5, 0.5, num_points), torch.linspace(-0.5, 0.5, num_points)
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
        energy3 = (ncn.expectation() ** 2 + ncn.variance()).sum()
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

        torch.testing.assert_allclose(
            energy1, energy2, msg="Not enough data points in spline or error while computing the derivatives"
        )
        torch.testing.assert_allclose(
            energy1, energy3, msg="Error with the derivative or with Non Central Nakagami function"
        )
