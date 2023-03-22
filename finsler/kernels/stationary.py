# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Pablo Moreno-Munoz
# Dept. of Signal Processing and Com. -- (pmoreno@tsc.uc3m.es, pabmo@dtu.dk)
# Universidad Carlos III de Madrid

import numpy as np
import torch

from finsler.kernels.kernel import Kernel


def squared_distance_old(x1, x2=None):
    """
    Given points x1 [n1 x d1] and x2 [n2 x d2], return a [n1 x n2] matrix with
    the pairwise squared distances between the points.
    Entry (i, j) is sum_{j=1}^d (x_1[i, j] - x_2[i, j]) ^ 2
    """
    if x2 is None:
        return squared_distance(x1, x1)

    x1s = x1.pow(2).sum(1, keepdim=True)
    x2s = x2.pow(2).sum(1, keepdim=True)

    r2 = x1s + x2s.t() - 2.0 * x1 @ x2.t()

    # Prevent negative squared distances using torch.clamp
    # NOTE: Clamping is for numerics.
    # This use of .detach() is to avoid breaking the gradient flow.
    return r2 - (torch.clamp(r2, max=0.0)).detach()


def squared_distance(x1, x2=None):
    """
    Given points x1 [n1 x d1] and x2 [n2 x d2], return a [n1 x n2] matrix with
    the pairwise squared distances between the points.
    Entry (i, j) is sum_{j=1}^d (x_1[i, j] - x_2[i, j]) ^ 2
    """
    if x2 is None:
        x2 = x1
    if x1.dim() == 2:
        x1 = torch.unsqueeze(x1, 0)
    if x2.dim() == 2:
        x2 = torch.unsqueeze(x2, 0)

    if x1.shape[0] != x2.shape[0]:
        x2 = x2.repeat(x1.shape[0], 1, 1)  # batch size should be the same between x1 and x2

    # sum over the num of dimensions
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_pad = torch.ones_like(x2_norm)

    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    # x1s = x1.pow(2).sum(2, keepdim=True)
    # x2s = x2.pow(2).sum(2, keepdim=True)
    # r2 = x1s + x2s.transpose(1, 2) -2.0 * torch.bmm(x1, x2.transpose(1, 2))
    res = torch.clamp(res, min=0.0).detach()

    return res


class Stationary(Kernel):
    """
    -- * Stationary

        Description:        Class for Stationary Kernel
        ----------
        Parameters
        ----------
        - length_scale:     float / lengthscale hyperparameter
        - variance:         float / variance hyperparameter
        - input_dim:        int / dimensionality of X
        - ARD:              bool / automatic relevant determination? a pair of hyperparameters per dim of X
        - fit_hyp:          bool / trainable hyperparams?
    """

    def __init__(self, variance=None, length_scale=None, input_dim=None, ARD=False, fit_hyp=True):
        super().__init__(input_dim)

        if input_dim is None:
            self.input_dim = 1
        else:
            self.input_dim = input_dim

        self.ARD = ARD  # Automatic relevance determination
        # Length-scale/smoothness of the kernel -- l
        if self.ARD:
            if length_scale is None:
                log_ls = torch.log(torch.tensor(0.1)) * torch.ones(self.input_dim)
            else:
                log_ls = torch.log(length_scale) * torch.ones(self.input_dim)
        else:
            if length_scale is None:
                log_ls = torch.log(torch.tensor(0.1)) * torch.ones(1)
            else:
                log_ls = torch.log(length_scale) * torch.ones(1)

        # Variance/amplitude of the kernel - /sigma
        if variance is None:
            log_variance = torch.log(torch.tensor(2.0))
        else:
            log_variance = torch.log(variance)

        if fit_hyp:
            self.log_length_scale = torch.nn.Parameter(log_ls, requires_grad=True)
            self.log_variance = torch.nn.Parameter(log_variance * torch.ones(1), requires_grad=True)
            self.register_parameter("length_scale", self.log_length_scale)
            self.register_parameter("variance", self.log_variance)
        else:
            self.log_length_scale = torch.nn.Parameter(log_ls, requires_grad=fit_hyp)
            self.log_variance = torch.nn.Parameter(log_variance * torch.ones(1), requires_grad=fit_hyp)
            self.register_parameter("length_scale", self.log_length_scale)
            self.register_parameter("variance", self.log_variance)

    def squared_dist(self, X, X2):
        """
        Returns the SCALED squared distance between X and X2.
        """
        length_scale = torch.exp(self.log_length_scale).abs().clamp(min=0.01, max=10.0)  # minimum enforced
        return squared_distance(X / length_scale, X2 / length_scale)

    def Kdiag(self, X):
        variance = torch.abs(torch.exp(self.variance))
        return variance.expand(X.size(0))
