import argparse

import matplotlib.pyplot as plt
import numpy as np

from examples.indicatrices import Indicatrices, Tensor, automated_scaling
from finsler.visualisation.indicatrices import contour_high_dim, contour_test


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outDir", default="latent_distances_finsler/plots/high_dimensions/", type=str)
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--q", default=2, type=int)
    parser.add_argument("--D", default=5, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    opts = parser.parse_args()
    return opts


def matrix_histogram(J, G):
    plt.hist(J[0, 0, :], color="blue", label="gaussian", alpha=0.4)
    plt.hist(G[0, 0, :], color="red", label="whishart", alpha=0.4)
    plt.legend()
    plt.show()


def test_simulation():
    np.random.seed(opts.seed)
    loc = 0
    scale = np.random.random()
    size = (opts.q, opts.D, opts.num_samples)
    ncgaussian = np.random.normal(loc, scale, size)
    ncwihsart = np.einsum("ijk,mjk->imk", ncgaussian, ncgaussian)

    exp_whishart = np.mean(ncwihsart, axis=2)
    cov = exp_whishart / opts.D
    mean = np.empty(cov.shape)

    alpha = automated_scaling(exp_whishart)
    vectors = np.linspace(-1.1 * alpha, 1.1 * alpha, 64)
    indicatrix = Indicatrices(3, vectors)

    finsler_sim = indicatrix.simulated_finsler(ncwihsart)
    riemann_sim = indicatrix.simulated_riemann(ncwihsart)
    finsler_expl = indicatrix.explicit_finsler(cov, mean)
    riemann_expl = indicatrix.explicit_riemann(cov, mean)

    contour_test(
        finsler_sim,
        riemann_sim,
        finsler_expl,
        riemann_expl,
        opts.outDir,
        # title="mean: {}, cov: {}".format(mean, cov),
        name="check_{}".format(opts.seed),
    )


if __name__ == "__main__":
    opts = get_args()

    dims = [3, 5, 50]
    finsler_indicatrices = np.empty((len(dims), 64, 64))
    riemann_indicatrices = np.empty((len(dims), 64, 64))
    size = (opts.q, dims[-1], opts.num_samples)  # J.T used here for compute

    for opts.seed in range(21):
        np.random.seed(opts.seed)  # 12 #31
        # simulate central gaussian and central wishart
        loc = np.random.randn()
        scale = np.random.random()
        ncgauss = np.random.normal(loc, scale, size)

        for idx, dim in enumerate(dims):
            ncgauss_tronc = ncgauss[:, :dim, :]
            print("dimension: {}".format(ncgauss_tronc.shape))
            ncwihsart = np.einsum("ijk,mjk->imk", ncgauss_tronc, ncgauss_tronc)

            tensor = Tensor()
            exp_whishart = np.mean(ncwihsart, axis=2)
            alpha = automated_scaling(exp_whishart)

            vectors = np.linspace(-1.1 * alpha, 1.1 * alpha, 64)
            indicatrix = Indicatrices(3, vectors)
            finsler_indicatrices[idx] = indicatrix.simulated_finsler(ncwihsart)
            riemann_indicatrices[idx] = indicatrix.simulated_riemann(ncwihsart)

        contour_high_dim(
            finslers=finsler_indicatrices,
            riemanns=riemann_indicatrices,
            dims=dims,
            out_dir=opts.outDir,
            # title="mean: {}, cov: {}".format(mean, cov),
            name="high_dim_seed_{}".format(opts.seed),
        )
        print("high_dim_seed_{} was saved!".format(opts.seed))
