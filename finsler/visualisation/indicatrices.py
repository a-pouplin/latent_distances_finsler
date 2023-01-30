import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse


def ellipse_params(matrix):
    """Get the params of an ellipse for the matplotlib func: ellpse_draw.
    ---
    input: 2x2 psd matrix
    output: semi-axis (width and height), angles (in degrees)"""
    eigvalues, eigvectors = np.linalg.eig(matrix)
    idx = np.argsort(eigvalues)
    eigvalues, eigvectors = eigvalues[idx], eigvectors[:, idx]
    height, width = 1 / np.sqrt(eigvalues[1]), 1 / np.sqrt(eigvalues[0])
    (cos_theta, sin_theta) = eigvectors[:, 1]
    theta = np.arctan(sin_theta / cos_theta)
    return 2 * width, 2 * height, -np.rad2deg(theta)


def ellipse_draw(matrix, center, scale, **kwargs):
    """Get matplotlib patch of an ellipse draw from a psd 2x2 matrix."""
    width, height, angle = ellipse_params(matrix)
    return Ellipse(center, width=scale * width, height=scale * height, angle=angle, fill=False, **kwargs)


def PolyArea(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def contour_high_dim(finslers, riemanns, dims, out_dir, name, title=None, legend=True):
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    norm = plt.Normalize()
    colormap = sns.color_palette("husl", len(dims))
    csf = []
    csr = []
    for idx, dim in enumerate(dims):
        axs.contour(finslers[idx], (1,), cmap="plasma", linewidths=1, alpha=0.5)
        axs.contour(riemanns[idx], (1,), cmap="plasma", linewidths=1, alpha=0.5, linestyles=[(0, (1, 5))])
        csf.append(axs.contour(finslers[idx], (1,), color=colormap[idx], linewidths=1, alpha=0.5))
        csr.append(
            axs.contour(riemanns[idx], (1,), color=colormap[idx], linewidths=1, alpha=0.5, linestyles=[(0, (1, 5))])
        )
    csr = axs.contour(riemanns[idx], (1,), colors="k", linewidths=1, linestyles=[(0, (1, 5))])

    if legend:
        labels = ["dim: {}".format(dim) for dim in dims]
        artists = []
        for i, label in enumerate(labels):
            artists.append(csf[i].legend_elements()[0][0])
            artists.append(csr[i].legend_elements()[0][0])
        # artists.append(csr.legend_elements()[0][0])
        # labels.append("Riemann, dim: {}".format(dims[-1]))
        axs.legend(
            handles=artists,
            labels=labels,
            prop={"size": 6},
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, -0.05),
        )

    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal")
    axs.set_title(title)
    axs.set(frame_on=False)
    fig.savefig(os.path.join(out_dir, "{}.svg".format(name)), dpi=fig.dpi, bbox_inches="tight")


def contour_bounds(finsler, riemann, lower, out_dir, name, title=None, legend=False):
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    cs = []
    cs.append(axs.contour(finsler, (1,), colors="tab:orange", linewidths=1))
    cs.append(axs.contour(riemann, (1,), colors="tab:purple", linewidths=1))
    cs.append(axs.contour(lower, (1,), colors="tab:green", linewidths=1))

    if legend:
        labels = ["Finsler", "Riemann", "Lower bound"]
        artists = []
        for i, label in enumerate(labels):
            artists.append(cs[i].legend_elements()[0][0])
        axs.legend(
            handles=artists,
            labels=labels,
            prop={"size": 6},
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, -0.05),
        )
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal")
    axs.set_title(title)
    axs.set(frame_on=False)
    fig.savefig(os.path.join(out_dir, "{}.svg".format(name)), dpi=fig.dpi, bbox_inches="tight")


def contour_test(finsler_sim, riemann_sim, finsler_expl, riemann_expl, out_dir, name, title=None):
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    axs.contour(finsler_sim, (1,), colors="tab:orange", linewidths=2)
    axs.contour(riemann_sim, (1,), colors="tab:blue", linewidths=2)
    axs.contour(finsler_expl, (1,), colors="tab:red", linewidths=2, linestyles="dashed")
    axs.contour(riemann_expl, (1,), colors="tab:green", linewidths=2, linestyles="dashed")
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc="tab:orange", ec="white", alpha=0.7, linewidth=4)
    proxy2 = plt.Rectangle((0, 0), 1, 1, fc="tab:blue", ec="white", alpha=0.7, linewidth=4)
    proxy3 = plt.Rectangle((0, 0), 1, 1, fc="tab:red", ec="white", alpha=0.7, linewidth=4)
    proxy4 = plt.Rectangle((0, 0), 1, 1, fc="tab:green", ec="white", alpha=0.7, linewidth=4)
    axs.legend(
        handles=[proxy1, proxy2, proxy3, proxy4], labels=["Finsler sim", "Riemann sim", "Finsler expl", "Riemann expl"]
    )
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal")
    axs.set_title(title)
    axs.set(frame_on=False)
    fig.savefig(os.path.join(out_dir, "{}.png".format(name)), dpi=fig.dpi, bbox_inches="tight")
