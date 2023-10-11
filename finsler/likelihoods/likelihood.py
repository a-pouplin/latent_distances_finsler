# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Pablo Moreno-Munoz
# Dept. of Signal Processing and Com. -- (pmoreno@tsc.uc3m.es, pabmo@dtu.dk)
# Universidad Carlos III de Madrid

import numpy as np
import torch


class Likelihood(torch.nn.Module):
    """
    Base class for likelihoods
    """

    def __init__(self):
        super(Likelihood, self).__init__()

    def gh_points(self, T=20):
        # Gaussian-Hermite Quadrature points
        gh_p, gh_w = np.polynomial.hermite.hermgauss(T)
        gh_p, gh_w = torch.from_numpy(gh_p), torch.from_numpy(gh_w)
        return gh_p, gh_w
