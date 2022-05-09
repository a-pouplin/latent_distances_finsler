import argparse

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import scipy.special
from matplotlib.path import Path

from finsler.utils.helper import psd_matrix
from finsler.visualisation.indicatrices import contour_riemann


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outDir", default="plots/indicatrices/", type=str)
    opts = parser.parse_args()
    return opts


def PolyArea(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class NonCentralNakagami:
    """Non central Nakagami distribution computed for data points Jx.
    inputs:
        - mean: the mean of Jaccobian (matrix of size: DxD)
        - cov: the covariance of Jaccobian (matrix of size: DxD)
        - x: latent vectors (matrix of size: 1xD)
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
        omega, D = mu / (var + 1e-10), self.D
        term_gamma = scipy.special.gamma((D + 1) / 2) / scipy.special.gamma(D / 2)
        term_hyp1f1 = scipy.special.hyp1f1((-1 / 2), (D / 2), -1 / 2 * omega)
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

    def finsler2(self, image, vectors):
        figc, axc = plt.subplots(1, 1)
        vec_size = len(vectors)
        vec_values = (np.max(vectors) - np.min(vectors)) ** 2
        cs = axc.contour(image, (1,))
        plt.close(figc)
        sgm = cs.allsegs[0][0]
        # sgm = cs.collections[0].get_segments()[0]
        eps = 1e-2
        assert (abs(sgm[0, 0] - sgm[-1, 0]) < eps) and (abs(sgm[0, 1] - sgm[-1, 1]) < eps), "Contour not closed!"
        pp = cs.collections[0].get_paths()[0].vertices / vec_size  # - [0.5,0.5] + points[nn]
        codes = Path.CURVE3 * np.ones(len(pp), dtype=Path.code_type)
        codes[0] = codes[-1] = Path.MOVETO
        path = Path(pp, codes)
        volume_indicatrix = PolyArea(path.vertices) * (vec_values)
        return np.pi / volume_indicatrix

    def finsler(self, image, vectors):
        return np.pi / self.indicatrix(image, vectors)

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
    output: DxD positive definite matrix.
    """

    def riemann(self, cov, mean, D):
        # expectation of M, with M ~ W_q(D, cov, inv(cov) @ mean)
        return D * cov + mean

    def fundamental(self, v):
        nakagami = NonCentralNakagami(cov=cov, mean=mean, D=D)
        FF = nd.Hessian(lambda x: nakagami.expectation(x) ** 2)
        return 1 / 2 * FF(v)


def indicatrix(cov, mean, D, vectors):
    """Draw the indicatrix of functions using a range of vectors.
    Let's have f: R^q -> R^D, a stochastic mapping,
    with J = Jacobian(f).
    ---
    inputs: - cov: qxq covariance matrix, with cov = var[J]
            - mean: qxq matrix, with mean = E[J].T@E[J]
            - D: dimension of the observational space:
            - vectors: range of vectors that helps compute the functions
    """
    size = len(vectors)
    finsler = np.empty((size, size))
    nakagami = NonCentralNakagami(cov=cov, mean=mean, D=D)
    for i1, y1 in enumerate(vectors):
        for i2, y2 in enumerate(vectors):
            y = np.array([y1, y2])  # random vectors
            finsler[i1, i2] = nakagami.expectation(y)
    return finsler


def indicatrix_riemann(cov, mean, D, vectors):
    """Draw the indicatrix of functions using a range of vectors.
    Let's have f: R^q -> R^D, a stochastic mapping, with J = Jacobian(f).
    ---
    inputs: - cov: qxq covariance matrix, with cov = var[J]
            - mean: qxq matrix, with mean = E[J].T@E[J]
            - D: dimension of the observational space:
            - vectors: range of vectors that helps compute the functions
    """
    size = len(vectors)
    riemann = np.empty((size, size))
    tensor = Tensor()
    metric = tensor.riemann(cov, mean, D)
    for i1, y1 in enumerate(vectors):
        for i2, y2 in enumerate(vectors):
            y = np.array([y1, y2])  # random vectors
            riemann[i1, i2] = y @ metric @ y.T
    return riemann


def automated_scaling2(metric):
    """scale the vector to compute the indicatrix"""
    eigvalues, eigvectors = np.linalg.eig(metric)
    long_size, short_size = 1 / np.sqrt(np.min(eigvalues)), 1 / np.sqrt(np.max(eigvalues))
    (cos_theta, sin_theta) = eigvectors[:, 0]
    theta = -np.rad2deg(np.arctan(sin_theta / cos_theta))
    if (theta % 180) < 45 or (theta % 180) > 135:
        return long_size, short_size
    elif 45 < (theta % 180) < 135 or (theta % 180) > 135:
        return short_size, long_size
    else:
        print("error with angle")


def automated_scaling(metric):
    """scale the vector to compute the indicatrix"""
    eigvalues, _ = np.linalg.eig(metric)
    return 1 / np.sqrt(np.min(eigvalues))


if __name__ == "__main__":

    upper_bounds = np.empty((100,))
    for seed_id in range(10):
        np.random.seed(seed_id)  # 12 #31
        opts = get_args()

        # inputs that define the random metric M ~ W_q(D, cov, inv(cov) @ mean)
        D = 3
        # mu = np.array([[-0.2747,  0.8382, -0.8249],
        #                [-0.9508, -0.6330,  0.0297]])
        # cov = np.array([[ 2.0370e-05,  1.2125e-05],
        #                 [ 6.0965e-05,  3.2783e-07]])
        #
        # cov = nearPD(cov)
        # # print(np.all(np.linalg.eigvals(cov) > 0))
        # mean = mu@np.transpose(mu)
        # mean = np.array([[1.4402521, 0.13207239],[0.13207239, 0.22419274]])
        # cov = np.array([[0.0010255, 0.00054871], [0.00054871, 0.00087961]])
        mean = psd_matrix(1e-6 + np.random.rand(2))
        cov = psd_matrix(1e-6 + np.random.rand(2))  # WARNING: should be q, not D !

        # compute finsler indicatrix
        # vectors = np.linspace(-5, 5, 32)

        tensor = Tensor()
        riemann = tensor.riemann(cov, mean, D)
        alpha = automated_scaling(riemann)
        # vectors_x = np.linspace(-1.1*alpha_x, 1.1*alpha_x, 32)
        # vectors_y = np.linspace(-1.1*alpha_y, 1.1*alpha_y, 32)
        # vectors = (vectors_x, vectors_y)
        vectors = np.linspace(-1.1 * alpha, 1.1 * alpha, 32)
        finsler_indicatrix = indicatrix(cov, mean, D, vectors)
        # fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
        # axs.contour(finsler_indicatrix, (1,), colors='orange', linewidths=2, alpha=0.7)
        # axs.imshow(finsler_indicatrix)
        # plt.show()
        # raise
        riemann_indicatrix = indicatrix_riemann(cov, mean, D, vectors)
        # contour_test(finsler_indicatrix, riemann_indicatrix, riemann, vectors, opts.outDir)
        # raise

        volume = Volume()
        # finsler_vol = volume.finsler(finsler_indicatrix, vectors)
        finsler_vol2 = volume.finsler2(finsler_indicatrix, vectors)
        riemann_vol = volume.riemann(cov, mean, D)
        riemann_vol2 = volume.finsler2(riemann_indicatrix, vectors)

        contour_riemann(
            finsler_indicatrix,
            riemann_indicatrix,
            vectors,
            opts.outDir,
            title="mean: {}, cov: {}".format(mean, cov),
            name="riemann_{}".format(seed_id),
        )

        print("finsler:", finsler_vol2)
        print("riemann:", riemann_vol2)
        # upper_bound_vol = volume.upper_bound(cov, mean, D)
        # ratio_vol = (rieman_vol-finsler_vol)/rieman_vol
        # assert (upper_bound_vol-ratio_vol)>0, 'Issue with {}'.format(seed_id)
        # print('{}: True'.format(seed_id))
        # # print('seed:{}, ratio:{:.2f}, upper bound -
        # ratio :{:.2f}'.format(seed_id, ratio_vol, upper_bound_vol-ratio_vol))
        # upper_bounds[seed_id] = upper_bound_vol
    # plt.hist(ratio_vol, 20, density=True, alpha=0.75)
    # plt.show()
