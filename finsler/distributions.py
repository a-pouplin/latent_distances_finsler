import warnings

import numpy as np
import torch
from scipy.special import gamma, hyp1f1
from torch.autograd import Function


class H1f1Gradients_old(Function):
    """Confluent hypergeometric function 1F1.
    Note:     a and b should be scalar
              b should be different to zero
              z should be negative and in float64

    Defintion: https://en.wikipedia.org/wiki/Confluent_hypergeometric_function
    Gradient: d/dx(1F1(a, b, x)) = (a 1F1(a + 1, b + 1, x))/b
    """

    @staticmethod
    def forward(ctx, a, b, z):
        # inputs in torch
        ctx.save_for_backward(a, b, z)
        z_np = z.detach().numpy()
        a = a.detach().numpy()
        b = b.detach().numpy()
        return torch.from_numpy(hyp1f1(a, b, z_np)).to(z)

    @staticmethod
    def backward(ctx, grad_output):
        a, b, z = ctx.saved_tensors
        if ctx.needs_input_grad[2]:
            a, b, z = a.detach().numpy(), b.detach().numpy(), z.detach().numpy()
            grad_hyp1f1 = (a / b) * hyp1f1(a + 1, b + 1, z)
            grad_hyp1f1 = torch.from_numpy(grad_hyp1f1).to(grad_output)
            grad_input = grad_output * grad_hyp1f1
        return None, None, grad_input


class NonCentralNakagami_old:
    """Non central Nakagami distribution computed for data points z.

    Disclaimer: Do not use this function is D > 150,
    scipy.Hyp1f1 becomes unstable, and would return 0.

    inputs:
        - var: the variance of z (vector of size: N)
        - mu: the mean of z (matrix of size: NxD)
    source:
        S. Hauberg, 2018, "The non-central Nakagami distribution"
    """

    def __init__(self, mu, var):
        self.D = mu.shape[-1]
        if self.D > 150:
            warnings.warn("D > 150, scipy.Hyp1f1 and scipy.gamma become unstable. Approximation is used.")
        self.var = var
        # self.omega = (mu**2).sum(1) / var
        self.omega = (mu**2).sum(dim=-1) / var

    def expectation(self):
        # eq 2.9. expectation = E[|z|], when z ~ N(mu, var)
        var, D, omega = self.var, self.D, self.omega
        const = np.sqrt(2)
        if D > 150:
            term_gamma = np.sqrt(D) / np.sqrt(2)
        else:
            term_gamma = gamma((D + 1) / 2) / gamma(D / 2)

        term_hyp1f1 = H1f1Gradients.apply(torch.tensor(-1 / 2), torch.tensor(D / 2), -1 / 2 * omega)
        expectation = torch.sqrt(var) * const * term_gamma * term_hyp1f1
        return expectation

    def variance(self):
        # eq 2.11. variance = var[|z|], when z ~ N(mu, var)
        var, D, omega = self.var, self.D, self.omega
        term_gamma = gamma((D + 1) / 2) / gamma(D / 2)
        term_hyp1f1 = H1f1Gradients.apply(torch.tensor(-1 / 2), torch.tensor(D / 2), -1 / 2 * omega)
        variance = var * (omega + D - 2 * (term_gamma * term_hyp1f1) ** 2)
        return variance


class H1f1Gradients(Function):
    """Confluent hypergeometric function 1F1.
    Note:     a and b should be scalar
              b should be different to zero
              z should be negative and in float64 # z tensor ?

    Defintion: https://en.wikipedia.org/wiki/Confluent_hypergeometric_function
    Gradient: d/dx(1F1(a, b, x)) = (a 1F1(a + 1, b + 1, x))/b
    """

    @staticmethod
    def forward(ctx, a: torch.FloatTensor, b: torch.FloatTensor, z: torch.FloatTensor):
        z = z.double()
        assert a.type() == "torch.FloatTensor", "a should be torch.float"
        assert b.type() == "torch.FloatTensor", "b should be torch.float"
        assert a.dim() == 0 and b.dim() == 0, "a and b should be scalar"
        assert b != 0, "b should be different to zero"
        assert torch.all(torch.le(z, 0)), f"All elements should be negative but found: {z.max()}"

        # inputs in torch
        ctx.save_for_backward(a, b, z)
        z_np = z.detach().numpy()
        a = a.detach().numpy()
        b = b.detach().numpy()
        return torch.from_numpy(hyp1f1(a, b, z_np)).to(z)

    @staticmethod
    def backward(ctx, grad_output):
        a, b, z = ctx.saved_tensors
        if ctx.needs_input_grad[2]:
            a, b, z = a.detach().numpy(), b.detach().numpy(), z.detach().numpy()
            grad_hyp1f1 = (a / b) * hyp1f1(a + 1, b + 1, z)
            grad_hyp1f1 = torch.from_numpy(grad_hyp1f1).to(grad_output)
            grad_input = grad_output * grad_hyp1f1
        return None, None, grad_input


class NonCentralNakagami:
    """Non central Nakagami distribution computed for data points z.
    working with batches of data points

    Disclaimer: Do not use this function is D > 150,
    scipy.Hyp1f1 becomes unstable, and would return 0.

    inputs:
    :param mu: (bs, num_data, dim_data)
    :param var: (bs, num_data)

    return:
    :param expectation: (bs, num_data)
    :param variance: (bs, num_data)

    source:
        S. Hauberg, 2018, "The non-central Nakagami distribution"
    """

    def __init__(self, mu: torch.FloatTensor, var: torch.FloatTensor):
        """
        :param mu: (bs, num_data, dim_data)
        :param var: (bs, num_data)
        """
        # work with batches of data points
        if mu.dim() == 2:
            mu = mu.unsqueeze(0)
        if var.dim() == 1:
            var = var.unsqueeze(0)
        var = torch.abs(var)
        # assert torch.all(torch.gt(var, 0)), f"All elements should be positive but found: {var.min()}"
        self.dim_data = mu.shape[-1]  # dim_data
        self.var = var  # (bs, num_data)
        self.omega = (mu**2).sum(dim=-1) / var  # (bs, num_data)
        if self.dim_data > 150:
            warnings.warn(f"dim_data: {self.dim_data} > 150, scipy.gamma becomes unstable. Approximation is used.")
            self.term_gamma = np.sqrt(self.dim_data) / np.sqrt(2)
        else:
            self.term_gamma = gamma((self.dim_data + 1) / 2) / gamma(self.dim_data / 2)
        self.term_hyp1f1 = H1f1Gradients.apply(
            torch.tensor(-1 / 2), torch.tensor(self.dim_data / 2), -1 / 2 * self.omega
        )

    def expectation(self):
        # eq 2.9. expectation = E[|z|], when z ~ N(mu, var)
        expectation = torch.sqrt(self.var) * np.sqrt(2) * self.term_gamma * self.term_hyp1f1
        return expectation

    def variance(self):
        # eq 2.11. variance = var[|z|], when z ~ N(mu, var)
        variance = self.var * (self.omega + self.dim_data - 2 * (self.term_gamma * self.term_hyp1f1) ** 2)
        assert variance.min() >= 0, f"variance should be positive but found: {variance.min()}"
        return variance
