# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Pablo Moreno-Munoz
# Dept. of Signal Processing and Com. -- (pmoreno@tsc.uc3m.es, pabmo@dtu.dk)
# Universidad Carlos III de Madrid

import numpy as np
import torch

from finsler.kernels.stationary import Stationary


class RBF(Stationary):
    """
    -- * RBF: Radial Basis Function or Squared Exponential / Gaussian Kernel

        Description:    Class for the RBF Kernel
        ----------
        Parameters
        ----------
        - length_scale:     float / lengthscale hyperparameter
        - variance:         float / variance hyperparameter
        - input_dim:        int / dimensionality of X
        - ARD:              bool / automatic relevant determination? a pair of hyperparameters per dim of X
        - fit_hyp:          bool / trainable hyperparams?
        - jitter:           float / jitter for positive-definiteness
    """

    def __init__(self, variance=None, length_scale=None, jitter=1e-5, input_dim=None, ARD=False, fit_hyp=True):

        super().__init__(length_scale=length_scale, variance=variance, input_dim=input_dim, ARD=ARD, fit_hyp=fit_hyp)
        self.jitter = jitter

    def K(self, X, X2=None):
        variance = torch.exp(self.log_variance).abs().clamp(min=0.0, max=np.inf)
        r2 = torch.clamp(self.squared_dist(X, X2), min=0.0, max=np.inf)
        K = variance * torch.exp(-r2 / 2.0)

        return K

    def jitchol(self, K, max_tries=5):
        # Cholesky decomposition + jitter // Inspired in the
        # Assure that is PSD
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            diagK = torch.diag(K)
            if torch.any(diagK <= 0.0):
                raise ValueError("Non-positive diagonal elements in K")

            if K.shape[0] == K.shape[1]:
                jitter = torch.mean(diagK) * 1e-6
                num_tries = 1
                while num_tries <= max_tries and torch.isfinite(jitter):
                    try:
                        L = torch.linalg.cholesky(K + torch.eye(K.shape[0]) * jitter)
                        return L
                    except:
                        jitter *= 10
                    finally:
                        num_tries += 1
                raise ValueError("Non-positive definite, even with jitter")
        return L
