import argparse
import os

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from matplotlib.path import Path
from scipy.special import gamma, hyp1f1

from finsler.utils.helper import create_folder, psd_matrix
from finsler.visualisation.indicatrices import PolyArea, contour_bounds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outDir", default="plots/bound_comparison", type=str)
    parser.add_argument("--seed", default=7, type=int)
    opts = parser.parse_args()
    return opts


class NonCentralNakagami:
    """Non central Nakagami distribution computed for data points Jx.
    inputs:
        - mean: E[J].T @ E[J] (matrix of size: qxq)
        - cov: the covariance of Jacobian (matrix of size: qxq)
        - x: latent vectors (matrix of size: 1xq)
    source:
        S. Hauberg, 2018, "The non-central Nakagami distribution"
    """

    def __init__(self, cov, mean, D):
        self.D = D
        self.cov = cov
        self.mean = mean

    def expectation(self, x):
        # eq 2.9. expectation = E[|Jx|], when J ~ N(mean, cov)
        var, mu = x.T @ self.cov @ x, x.T @ self.mean @ x
        omega = mu / (var + 1e-10)
        term_gamma = gamma((self.D + 1) / 2) / gamma(self.D / 2)
        term_hyp1f1 = hyp1f1((-1 / 2), (self.D / 2), -1 / 2 * omega)
        expectation = np.sqrt(var) * np.sqrt(2) * term_gamma * term_hyp1f1
        return expectation


class Volume:
    """Compute the riemannian and finslerian volume, where dimensions=2.
    ---
    inputs: - image: image that contains the indicatrix (F(vector)=1)
            - vectors: range of vectors used to create the image
            - cov: qxq covariance matrix, with cov = var[J]
            - mean: qxq matrix, with mean = E[J].T@E[J]
    output: Riemannian or Finslerian volumes.
    """

    def indicatrix(self, image, vectors):
        vec_size, vec_values = len(vectors), np.max(vectors) - np.min(vectors)
        scale = vec_values / vec_size
        indicatrix_vol = np.sum(1.0 * (image < 1))
        return indicatrix_vol * (scale**2)

    def indicatrix2(self, image, vectors):
        "other way to compute the volume of the indictarix based on the contour plot"
        figc, axc = plt.subplots(1, 1)
        vec_size = len(vectors)
        vec_values = (np.max(vectors) - np.min(vectors)) ** 2
        cs = axc.contour(image, (1,))
        plt.close(figc)
        sgm = cs.allsegs[0][0]
        # sgm = cs.collections[0].get_segments()[0]
        eps = 1e-2
        if (abs(sgm[0, 0] - sgm[-1, 0]) > eps) or (abs(sgm[0, 1] - sgm[-1, 1]) > eps):
            print("Contour not closed!")
            return np.nan
        pp = cs.collections[0].get_paths()[0].vertices / vec_size  # - [0.5,0.5] + points[nn]
        codes = Path.CURVE3 * np.ones(len(pp), dtype=Path.code_type)
        codes[0] = codes[-1] = Path.MOVETO
        path = Path(pp, codes)
        return PolyArea(path.vertices) * (vec_values)

    def hausdorff(self, image, vectors):
        "used to compute any volume: riemannian or finslerian"
        "can also be: np.pi / self.indicatrix2(image, vectors)"
        return np.pi / self.indicatrix2(image, vectors)

    def riemann(self, cov, mean, D):
        return np.sqrt(np.linalg.det(D * cov + mean))

    def upper_bound(self, cov, mean, D):
        omegas = np.empty((10,))
        q = 2
        for i, theta in enumerate(np.linspace(0, np.pi, 10)):
            e = np.array([np.cos(theta), np.sin(theta)])
            omegas[i] = (e.T @ mean @ e) / (e.T @ cov @ e)
        omega = np.min(omegas)
        upper_bound = 1 - (1 - 1 / (D + omega) + omega / (D + omega) ** 2) ** q
        return upper_bound


class Tensor:
    """Compute the riemannian and fundamental tensors.
    ---
    inputs: - D: number of dimension
            - cov: qxq covariance matrix, with cov = var[J]
            - mean: qxq matrix, with mean = E[J].T@E[J]
            - v: vector that will define the fundamnetal form
    output: qxq positive definite matrix.
    """

    def riemann(self, cov, mean, D):
        # expectation of M, with M ~ W_q(D, cov, inv(cov) @ mean)
        return D * cov + mean

    def fundamental(self, v, ov, mean, D):
        nakagami = NonCentralNakagami(cov=cov, mean=mean, D=D)
        FF = nd.Hessian(lambda x: nakagami.expectation(x) ** 2)
        return 1 / 2 * FF(v)

    def lower(self, cov, D):
        gamma_ratio = gamma((D + 1) / 2) / gamma(D / 2)
        alpha = 2 * (gamma_ratio**2)
        return alpha * cov


def compute_riemann_indicatrix(metric, vectors, size):
    """Draw the indicatrix for a psd metric tensor.
    ---
    inputs: - metric: qxq metric tensor, should be PSD
            - vectors: range of vectors that helps compute the functions
            - size: size of the image to draw the contour plot
    """
    indicatrix = np.empty((size, size))
    for i1, y1 in enumerate(vectors):
        for i2, y2 in enumerate(vectors):
            y = np.array([y1, y2])  # random vectors
            indicatrix[i1, i2] = y @ metric @ y.T
    return indicatrix


class Indicatrices:
    def __init__(self, D, vectors):
        """Draw the indicatrix of functions using a range of vectors.
        Let's have f: R^q -> R^D, a stochastic mapping,
        with J = Jacobian(f).
        ---
        inputs: - cov: qxq covariance matrix, with cov = var[J]
                - mean: qxq matrix, with mean = E[J].T@E[J]
                - D: dimension of the observational space:
                - vectors: range of vectors that helps compute the functions
        """
        self.vectors = vectors
        self.size = len(vectors)
        self.D = D

    def explicit_finsler(self, cov, mean):
        finsler = np.empty((self.size, self.size))
        nakagami = NonCentralNakagami(cov, mean, self.D)
        for i1, y1 in enumerate(self.vectors):
            for i2, y2 in enumerate(self.vectors):
                y = np.array([y1, y2])  # random vectors
                finsler[i1, i2] = nakagami.expectation(y)
        return finsler

    def explicit_riemann(self, cov, mean):
        tensor = Tensor()
        metric = tensor.riemann(cov, mean, self.D)
        return self.custom_riemann(metric)

    def explicit_lower(self, cov):
        tensor = Tensor()
        metric = tensor.lower(cov, self.D)
        return self.custom_riemann(metric)

    def simulated_riemann(self, random_metric):
        metric = np.mean(random_metric, axis=2)
        indicatrix = np.empty((self.size, self.size))
        for i1, y1 in enumerate(self.vectors):
            for i2, y2 in enumerate(self.vectors):
                y = np.array([y1, y2])  # random vectors
                indicatrix[i1, i2] = y @ metric @ y.T
        return indicatrix

    def simulated_finsler(self, random_metric):
        indicatrix = np.empty((self.size, self.size))
        num_sim = random_metric.shape[-1]
        for i1, y1 in enumerate(self.vectors):
            for i2, y2 in enumerate(self.vectors):
                y_ = np.array([y1, y2])  # random vectors
                y = np.repeat(y_[:, np.newaxis], num_sim, axis=1)
                My = np.einsum("ijk,jl->ik", random_metric, y)
                yMy = np.einsum("jk,jn->kn", y, My)[0, :] / num_sim
                np.allclose(yMy[0], y_ @ random_metric[:, :, 0] @ y_.T, rtol=1e-6)
                indicatrix[i1, i2] = np.mean(np.sqrt(yMy))
        return indicatrix

    def custom_riemann(self, metric):
        indicatrix = np.empty((self.size, self.size))
        for i1, y1 in enumerate(self.vectors):
            for i2, y2 in enumerate(self.vectors):
                y = np.array([y1, y2])  # random vectors
                indicatrix[i1, i2] = y @ metric @ y.T
        return indicatrix


def automated_scaling(metric):
    """scale the vector to compute the indicatrix"""
    eigvalues, _ = np.linalg.eig(metric)
    return 1 / np.sqrt(np.min(eigvalues))


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))

    folderpath = os.path.abspath(opts.outDir)
    create_folder(folderpath)
    print("--- figures saved in:", folderpath)

    for opts.seed in range(5):
        np.random.seed(opts.seed)
        mean = psd_matrix(1e-6 + np.random.rand(2))
        cov = psd_matrix(1e-6 + np.random.rand(2))

        tensor = Tensor()
        lower = tensor.lower(cov, 3)
        alpha = automated_scaling(lower)

        vectors = np.linspace(-1 * alpha, 1 * alpha, 64)
        indicatrix = Indicatrices(3, vectors)
        finsler_indicatrix = indicatrix.explicit_finsler(cov, mean)
        riemann_indicatrix = indicatrix.explicit_riemann(cov, mean)
        lower_indicatrix = indicatrix.explicit_lower(cov)

        print("indicatrix_{}".format(opts.seed))
        contour_bounds(
            finsler_indicatrix,
            riemann_indicatrix,
            lower_indicatrix,
            folderpath,
            name="indicatrix_{}".format(opts.seed),
            legend=True,
        )
