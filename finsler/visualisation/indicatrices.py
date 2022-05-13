import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def ellipse_params(matrix):
    """Get the params of an ellipse for the matplotlib func: ellpse_draw.
    ---
    input: 2x2 psd matrix
    output: semi-axis (width and height), angles (in degrees)"""
    eigvalues, eigvectors = np.linalg.eig(matrix)
    idx = np.argsort(eigvalues)
    eigvalues, eigvectors = eigvalues[idx], eigvectors[:, idx]
    # eigvalues = eigvalues[idx]
    height, width = 1 / np.sqrt(eigvalues[1]), 1 / np.sqrt(eigvalues[0])
    # (sin_theta, cos_theta) = eigvectors[:,1] # think of matrix == R(-theta)
    (cos_theta, sin_theta) = eigvectors[:, 1]
    theta = np.arctan(sin_theta / cos_theta)
    # if sin_theta > 0: # check the sign of the angle
    #     theta = np.arccos(cos_theta)
    # else:
    #     theta = -np.arccos(cos_theta)
    return 2 * width, 2 * height, -np.rad2deg(theta)


def ellipse_params_new(matrix):
    """Get the params of an ellipse for the matplotlib func: ellpse_draw.
    ---
    input: 2x2 psd matrix
    output: semi-axis (width and height), angles (in degrees)"""
    eigvalues, eigvectors = np.linalg.eig(matrix)
    idx = np.argsort(eigvalues)

    eigvalues, eigvectors = eigvalues[idx], eigvectors[:, idx]
    # eigvalues = eigvalues[idx]
    height, width = 1 / np.sqrt(eigvalues[1]), 1 / np.sqrt(eigvalues[0])
    # (sin_theta, cos_theta) = eigvectors[:,1] # think of matrix == R(-theta)
    (cos_theta, sin_theta) = eigvectors[:, 0]
    theta = np.arctan(sin_theta / cos_theta)
    # if sin_theta > 0: # check the sign of the angle
    #     theta = np.arccos(cos_theta)
    # else:
    #     theta = -np.arccos(cos_theta)
    return 2 * width, 2 * height, np.rad2deg(theta)


def ellipse_params2(matrix):  # only valid for 2x2 psd_matrix
    a, b, c = matrix[0, 0], matrix[0, 1], matrix[1, 1]
    theta2 = np.arctan(2 * b / (a - c))
    width_ = (c + a) + (a - c) / np.cos(theta2)
    height_ = (c + a) - (a - c) / np.cos(theta2)
    theta = theta2 / 2
    theta = theta * 180 / np.pi
    return 2 * np.sqrt(2 / width_), 2 * np.sqrt(2 / height_), theta


def ellipse_draw(matrix, center, scale, **kwargs):
    """Get matplotlib patch of an ellipse draw from a psd 2x2 matrix."""
    width, height, angle = ellipse_params(matrix)
    return Ellipse(center, width=scale * width, height=scale * height, angle=angle, fill=False, **kwargs)


def contour_fundamental(finsler, riemann, fundamental0, fundamental1, vectors, out_dir, with_eig="True"):
    scale = len(vectors) / (np.max(vectors) - np.min(vectors))
    center = (int(len(vectors) / 2), int(len(vectors) / 2))
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    axs.contour(finsler, (1,), colors="tab:orange", linewidths=1, linestyles="dashed")
    axs.add_patch(ellipse_draw(riemann, center, scale, linewidth=1, edgecolor="tab:blue", linestyle="dashed"))
    axs.add_patch(ellipse_draw(fundamental0, center, scale, linewidth=1, edgecolor="tab:purple"))
    axs.add_patch(ellipse_draw(fundamental1, center, scale, linewidth=1, edgecolor="tab:green"))
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc="tab:orange", ec="white", alpha=0.7, linewidth=4)
    proxy2 = plt.Rectangle((0, 0), 1, 1, fc="tab:blue", ec="white", alpha=0.7, linewidth=4)
    proxy3 = plt.Rectangle((0, 0), 1, 1, fc="tab:purple", ec="white", alpha=0.7, linewidth=4)
    proxy4 = plt.Rectangle((0, 0), 1, 1, fc="tab:green", ec="white", alpha=0.7, linewidth=4)
    axs.legend(handles=[proxy1, proxy2, proxy3, proxy4], labels=["finsler", "riemann", "fundamental0", "fundamental1"])
    if with_eig:
        _, eigvector = np.linalg.eig(riemann)
        # _, eigvec0 = np.linalg.eig(fundamental0)
        # _, eigvec1 = np.linalg.eig(fundamental1)
        # axs.arrow(*center, dx=10*eigvector[0,1], dy=-10*eigvector[1,1], color='tab:blue')
        axs.arrow(*center, dx=10 * eigvector[0, 0], dy=-10 * eigvector[1, 0], color="tab:purple")
        axs.arrow(*center, dx=10 * eigvector[0, 1], dy=-10 * eigvector[1, 1], color="tab:green")
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal")
    fig.savefig(os.path.join(out_dir, "{}.png".format("fundamentals")), dpi=fig.dpi)


def contour_riemann(finsler, riemann, vectors, out_dir, name, title=None):
    # scale = len(vectors) / (np.max(vectors) - np.min(vectors))
    # center = (int(len(vectors) / 2), int(len(vectors) / 2))
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    axs.contour(finsler, (1,), colors="tab:orange", linewidths=1)
    axs.contour(riemann, (1,), colors="tab:blue", linewidths=1)
    # axs.add_patch(ellipse_draw(riemann, center, scale, linewidth=1, edgecolor='tab:blue', linestyle='dashed'))
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc="tab:orange", ec="white", alpha=0.7, linewidth=4)
    proxy2 = plt.Rectangle((0, 0), 1, 1, fc="tab:blue", ec="white", alpha=0.7, linewidth=4)
    axs.legend(handles=[proxy1, proxy2], labels=["Finsler", "Riemann"])
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal")
    axs.set_title(title)
    fig.savefig(os.path.join(out_dir, "{}.png".format(name)), dpi=fig.dpi)


def contour_test_depreciated(finsler, riemann_test, riemann, scale, out_dir):
    normalise = len(scale) / (np.max(scale) - np.min(scale))
    center = (int(len(scale) / 2), int(len(scale) / 2))
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    axs.contour(finsler, (1,), colors="tab:orange", linewidths=1, linestyles="dashed")
    axs.contour(riemann_test, (1,), colors="tab:green", linewidths=1, linestyles="dashed")
    axs.add_patch(ellipse_draw(riemann, center, normalise, linewidth=1, edgecolor="tab:blue", linestyle="dashed"))

    _, eigvectors = np.linalg.eig(riemann)
    _, _, angle = ellipse_params(riemann)
    cos_theta, sin_theta = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))

    axs.arrow(*center, dx=-10 * eigvectors[0, 1], dy=10 * eigvectors[1, 1], color="tab:blue")
    axs.arrow(*center, dx=10 * cos_theta, dy=10 * sin_theta, color="tab:green")
    # axs.arrow(*center, dx=10*vr[0,1], dy=10*vr[0,0], color='tab:blue')

    legends = dict()
    legends["colors"] = ["tab:orange", "tab:green", "tab:blue"]
    legends["labels"] = ["finsler", "riemann_test", "riemann"]
    handles = [plt.Rectangle((0, 0), 1, 1, fc=color, ec="white", alpha=0.7, linewidth=4) for color in legends["colors"]]
    axs.legend(handles=handles, labels=legends["labels"])
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal", "box")
    fig.savefig(os.path.join(out_dir, "{}.png".format("test")), dpi=fig.dpi)


def contours(indicatrices, vector, scale, out_dir):
    # Get the figure in the right dimensions
    normalise = len(scale) / (np.max(scale) - np.min(scale))
    center = (int(len(scale) / 2), int(len(scale) / 2))
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

    # indiactrices plot
    axs.contour(indicatrices["values"][0], (1,), colors=indicatrices["colors"][0], linewidths=0.7, alpha=0.7)
    axs.contour(indicatrices["values"][1], (1,), colors=indicatrices["colors"][1], linewidths=0.5, alpha=0.7)
    for i in [2, 3]:
        axs.add_patch(
            ellipse_draw(
                indicatrices["values"][i],
                center,
                normalise,
                edgecolor=indicatrices["colors"][i],
                linewidth=1,
                linestyle="dashed",
            )
        )
    axs.arrow(*center, dx=50 * vector[0], dy=50 * vector[1], color=indicatrices["colors"][-1])

    # legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, ec="white", alpha=0.7, linewidth=4) for color in indicatrices["colors"]
    ]
    axs.legend(handles=handles, labels=indicatrices["labels"])
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal", "box")
    fig.savefig(os.path.join(out_dir, "{}.png".format("original")), dpi=fig.dpi)


def contour_angle(indicatrices, fundamentals, arrows, angles, scale, out_dir):
    # Get the figure in the right dimensions
    normalise = len(scale) / (np.max(scale) - np.min(scale))
    center = (int(len(scale) / 2), int(len(scale) / 2))
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

    axs.contour(
        indicatrices["values"][0], (1,), colors=indicatrices["colors"][0], linewidths=0.7, linestyles="dashed"
    )  # finsler plot
    axs.contour(
        indicatrices["values"][1], (1,), colors=indicatrices["colors"][1], linewidths=0.7, linestyles="dashed"
    )  # riemann plot
    colors = plt.cm.jet(np.linspace(0, 1, len(angles)))
    for i in range(len(angles)):
        axs.arrow(*center, dx=-50 * arrows[i, 1], dy=-50 * arrows[i, 0], color=colors[i])
        axs.add_patch(ellipse_draw(fundamentals[i], center, normalise, edgecolor=colors[i], linewidth=0.5))
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[i], ec="white", alpha=0.7, linewidth=4) for i in range(len(angles))
    ]
    axs.legend(handles=handles, labels=angles, prop={"size": 6})
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal", "box")
    fig.savefig(os.path.join(out_dir, "{}.png".format("angles")), dpi=fig.dpi)


def rotation_ellipse(image, angle, out_dir):
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    center = (int(image.shape[0] / 2), int(image.shape[1] / 2))
    axs.imshow(image, alpha=0.1)
    axs.add_patch(Ellipse(center, width=20, height=5, angle=angle, fill=False))
    cos_theta, sin_theta = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    print(cos_theta, sin_theta)
    axs.arrow(*center, dx=10 * cos_theta, dy=10 * sin_theta, color="tab:blue")
    axs.set_aspect("equal", "box")
    axs.set_xticks([])
    axs.set_yticks([])
    fig.savefig(os.path.join(out_dir, "{}.png".format("test_ellipse")), dpi=fig.dpi)


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


def PolyArea(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def contour_high_dim(finsler, riemann, out_dir, name, title=None):
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    axs.contour(finsler, (1,), colors="tab:orange", linewidths=2)
    axs.contour(
        riemann,
        (1,),
        colors="tab:blue",
        linewidths=2,
    )
    # axs.contour(custom, (1,), colors="tab:green", linewidths=2, linestyles='dashed')
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc="tab:orange", ec="white", alpha=0.7, linewidth=4)
    proxy2 = plt.Rectangle((0, 0), 1, 1, fc="tab:blue", ec="white", alpha=0.7, linewidth=4)
    # proxy3 = plt.Rectangle((0, 0), 1, 1, fc="tab:green", ec="white", alpha=0.7, linewidth=4)
    axs.legend(handles=[proxy1, proxy2], labels=["Finsler", "Riemann"])
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal")
    axs.set_title(title)
    axs.set(frame_on=False)
    fig.savefig(os.path.join(out_dir, "{}.png".format(name)), dpi=fig.dpi, bbox_inches="tight")


def contour_bounds(finsler, riemann, lower, out_dir, name, title=None, legend=False):
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    axs.contour(finsler, (1,), colors="tab:orange", linewidths=2)
    axs.contour(riemann, (1,), colors="tab:blue", linewidths=2)
    axs.contour(lower, (1,), colors="tab:green", linewidths=2)
    if legend:
        proxy1 = plt.Rectangle((0, 0), 1, 1, fc="tab:orange", ec="white", alpha=0.7, linewidth=4)
        proxy2 = plt.Rectangle((0, 0), 1, 1, fc="tab:blue", ec="white", alpha=0.7, linewidth=4)
        proxy3 = plt.Rectangle((0, 0), 1, 1, fc="tab:green", ec="white", alpha=0.7, linewidth=4)
        axs.legend(handles=[proxy1, proxy2, proxy3], labels=["Finsler", "Riemann", "Lower bound"])
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal")
    axs.set_title(title)
    axs.set(frame_on=False)
    fig.savefig(os.path.join(out_dir, "{}.png".format(name)), dpi=fig.dpi, bbox_inches="tight")


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
