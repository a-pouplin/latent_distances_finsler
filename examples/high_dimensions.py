import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from examples.bound_comparison import Indicatrices, Tensor, Volume, automated_scaling
from finsler.utils.helper import create_folder
from finsler.visualisation.indicatrices import contour_high_dim, contour_test


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outDir", default="plots/high_dimensions/", type=str)
    parser.add_argument("--q", default=2, type=int)  # latent dimension
    parser.add_argument("--num_samples", default=30, type=int)  # used for 'resolution'
    opts = parser.parse_args()
    return opts


def matrix_histogram(J, G):
    plt.hist(J[0, 0, :], color="blue", label="gaussian", alpha=0.4)
    plt.hist(G[0, 0, :], color="red", label="whishart", alpha=0.4)
    plt.legend()
    plt.show()


def test_simulation(D=5):
    loc = 0
    scale = np.random.random()
    size = (opts.q, D, opts.num_samples)
    ncgaussian = np.random.normal(loc, scale, size)
    ncwihsart = np.einsum("ijk,mjk->imk", ncgaussian, ncgaussian)

    exp_whishart = np.mean(ncwihsart, axis=2)
    cov = exp_whishart / D
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
        name="check",
    )


def plot_volume_relative(dims, volumes, title="plot_dims"):
    dims = np.array(dims)
    fig, ax = plt.subplots(1)
    ax.plot(dims, volumes, c="black", linewidth=1, linestyle="dashed", alpha=0.2)
    ax.plot(dims, np.mean(volumes, axis=1), c="black", linewidth=1, label="average difference of volume")
    ax.plot(dims, 1 / dims, c="red", linewidth=1, label=r"$f(x)=\frac{1}{x}$")

    ax.fill_between(
        dims,
        np.mean(volumes, axis=1) - 1.96 * np.std(volumes, axis=1),
        np.mean(volumes, axis=1) + 1.96 * np.std(volumes, axis=1),
        color="papayawhip",
        alpha=0.5,
    )
    ax.set_title(r"$\frac{V_G(x) - V_F(x)}{V_G(x)}$ with respect to the number of dimensions")
    ax.set_xlabel("dimensions")
    ax.set_ylabel("volume difference")
    ax.set_xlim(dims[0], dims[-1])
    ax.set_ylim(0, np.max(volumes))
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(opts.outDir, "{}.svg".format(title)), dpi=fig.dpi, bbox_inches="tight")
    plt.show()


def plot_volume_absolute(dims, finsler, riemann, lower, out_dir, name, title=None):
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.plot(dims, finsler.transpose(), colors="tab:orange")
    plt.plot(dims, riemann.transpose(), colors="tab:purple")
    # plt.plot(dims, lower.transpose(), colors="tab:green")


if __name__ == "__main__":
    opts = get_args()
    print("options: {}".format(opts))

    folderpath = os.path.abspath(opts.outDir)
    create_folder(folderpath)
    print("--- figures saved in:", folderpath)

    # # for more data points (relevant for plotting the relative volume ratio w/ dims)
    dims = np.geomspace(3, 100, 10).astype(int)
    seeds = range(4, 16)

    finsler_indicatrices = np.empty((len(dims), 64, 64))
    riemann_indicatrices = np.empty((len(dims), 64, 64))
    size = (opts.q, dims[-1], opts.num_samples)  # J.T used here for compute

    volume = Volume()
    finsler_volumes = np.empty((len(seeds), len(dims)))
    riemann_volumes = np.empty((len(seeds), len(dims)))
    rel_diff_volumes = np.empty((len(seeds), len(dims)))

    for ids, opts.seed in enumerate(seeds):
        print(f"Image {ids} of {len(seeds)}")
        np.random.seed(opts.seed)
        # simulate central gaussian and central wishart
        scale = np.random.uniform(low=0.0, high=10, size=1)
        loc = np.sqrt(scale)
        ncgauss = np.random.normal(loc, scale, size)

        for idd, dim in enumerate(dims):
            ncgauss_tronc = ncgauss[:, :dim, :]
            ncwihsart = np.einsum("ijk,mjk->imk", ncgauss_tronc, ncgauss_tronc)

            tensor = Tensor()
            exp_whishart = np.mean(ncwihsart, axis=2)
            alpha = automated_scaling(exp_whishart)

            vectors = np.linspace(-1.1 * alpha, 1.1 * alpha, 64)
            indicatrix = Indicatrices(3, vectors)
            finsler_indicatrices[idd] = indicatrix.simulated_finsler(ncwihsart)
            riemann_indicatrices[idd] = indicatrix.simulated_riemann(ncwihsart)

            finsler_volumes[ids, idd] = volume.hausdorff(finsler_indicatrices[idd], vectors)
            riemann_volumes[ids, idd] = volume.hausdorff(riemann_indicatrices[idd], vectors)

    contour_high_dim(
        finslers=finsler_indicatrices,
        riemanns=riemann_indicatrices,
        dims=[3, 5, 10, 50],
        out_dir=opts.outDir,
        # title="mean: {}, cov: {}".format(mean, cov),
        name="highdim_{}".format(opts.seed),
    )
    print("highdim_{} was saved!".format(opts.seed))

    vols = ((riemann_volumes - finsler_volumes) / riemann_volumes).transpose()
    plot_volume_relative(dims, vols, title="volumeratio")
