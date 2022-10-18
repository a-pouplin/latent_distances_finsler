# latent space visualisations for a trained GPLVM model

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import gaussian_kde, mode
from stochman.curves import CubicSpline
from stochman.geodesic import geodesic_minimizing_energy

from finsler.gplvm import gplvm
from finsler.utils.helper import pickle_load, pickle_save
from finsler.visualisation.latent import volume_heatmap


def get_args():
    parser = argparse.ArgumentParser()
    # manifold argments
    parser.add_argument("--num_geod", default=100, type=int)  # num of random geodesics to plot
    # data used
    parser.add_argument("--data", default="concentric_circles", type=str)  # sphere or qPCR or font or starfish
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--model_folder", default="trained_models/", type=str)
    parser.add_argument("--exp_folder", default="plots/length_distribution/", type=str)
    parser.add_argument("--title_model", default="_3_100", type=str)
    opts = parser.parse_args()
    return opts


def get_model(opts):
    # load previously training gplvm model
    model_folder = os.path.join(opts.model_folder, opts.data)
    model_saved = pickle_load(folder_path=model_folder, file_name="model{}.pkl".format(opts.title_model))
    model = model_saved["model"]
    Y = model_saved["Y"]
    X = model_saved["X"]

    # get Riemannian and Finslerian metric
    gplvm_riemann = gplvm(model, mode="riemannian")
    gplvm_finsler = gplvm(model, mode="finslerian")
    X = model.X_loc.detach().numpy()
    return gplvm_riemann, gplvm_finsler, X


def plot_latent_space(opts, gplvm_riemann, X, spline_riemann, spline_finsler):
    # plot latent space with geodesics
    fig = plt.figure(1)
    ax = plt.axes()
    ax, im, _ = volume_heatmap(ax, gplvm_riemann, X, mode="vol_riemann")
    for i in range(spline_riemann.shape[0]):
        ax.scatter(spline_riemann[i, ::2, 0], spline_riemann[i, ::2, 1], c="purple", s=0.5)
        ax.scatter(spline_finsler[i, 1::2, 0], spline_finsler[i, 1::2, 1], c="orange", s=0.5)
    ax.scatter(X[:, 0], X[:, 1], color="gray", s=1)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    fig.colorbar(im)
    plt.title("Geodesic in the latent space")
    # plt.show()
    return fig


def stats_geodesics(opts):
    gplvm_riemann, gplvm_finsler, X = get_model(opts)

    # Energy function computed with riemannian metric
    dim_latent = X.shape[1]

    optimizer = torch.optim.LBFGS
    max_iter = 20
    eval_grid = 20

    length_finsler = torch.empty((opts.num_geod, 1))
    length_riemann = torch.empty((opts.num_geod, 1))
    spline_finsler = torch.empty((opts.num_geod, eval_grid, dim_latent))
    spline_riemann = torch.empty((opts.num_geod, eval_grid, dim_latent))
    t = torch.linspace(0, 1, eval_grid)
    begin, ends = get_end_points(X)

    issue = 0
    for i in range(opts.num_geod):
        print("computing geodesic number {}/{} ...".format(i + 1, opts.num_geod))

        # curve_r, _ = gplvm_riemann.connecting_geodesic(begin, ends[:,i,:])
        # curve_f, _ = gplvm_finsler.connecting_geodesic(begin, ends[:,i,:]) #, init_curve=curve_r)

        print("\n")
        print("Riemannian")
        curve_r = CubicSpline(begin, ends[:, i, :], requires_grad=True)
        geodesic_minimizing_energy(curve_r, gplvm_riemann, optimizer, max_iter, eval_grid)

        print("\n")
        print("Finslerian")
        curve_f = CubicSpline(begin, ends[:, i, :], requires_grad=True)
        geodesic_minimizing_energy(curve_f, gplvm_finsler, optimizer, max_iter, eval_grid)

        with torch.no_grad():
            # length_riemann[i] = curve_r.euclidean_length()
            length_riemann[i] = gplvm_riemann.curve_length(curve_r(t))
            spline_riemann[i, :, :] = curve_r(t)
            # length_finsler[i] = curve_f.euclidean_length()
            length_finsler[i] = gplvm_finsler.curve_length(curve_f(t))
            spline_finsler[i, :, :] = curve_f(t)
        if length_riemann[i] < length_finsler[i]:
            diff1 = length_riemann[i] - length_finsler[i]
            diff2 = diff1 / length_riemann[i]
            print("Riemann - Finsler: {}; (Riemann - Finsler)/Riemann: {}".format(diff1, diff2))
            issue = issue + 1
    print("issues: {}/{}".format(issue, opts.num_geod))

    # plot latent space with geodesics
    print("to be saved in: {}".format(os.path.join(exp_folder, "latent{}.png".format(opts.title_model))))
    fig = plot_latent_space(opts, gplvm_riemann, X, spline_riemann, spline_finsler)
    fig.savefig(os.path.join(exp_folder, "latent{}.png".format(opts.title_model)), dpi=fig.dpi)
    print("saved ! as: {}".format(os.path.join(exp_folder, "latent{}.png".format(opts.title_model))))

    distribution = {
        "length_finsler": length_finsler,
        "length_riemann": length_riemann,
        "spline_finsler": spline_finsler,
        "spline_riemann": spline_riemann,
    }
    pickle_save(distribution, exp_folder, file_name="distribution{}_w_{}.pkl".format(opts.title_model, opts.num_geod))


def get_end_points(X, eps=0.25):
    # only get the end points of geodesic that are on the outer circles
    # and not too close from the start point
    norms = np.linalg.norm(X, axis=1)
    X_out = X[norms > 0.5]  # outer circles
    X_eps = X_out[np.linalg.norm(X_out - X_out[0], axis=1) > eps]  # not too close from the start point
    start, ends = torch.tensor([X_out[0]]), torch.tensor([X_eps])  # start and ends points
    return start, ends


def plt_dist(dist):
    lengths = np.concatenate((dist["length_riemann"], dist["length_finsler"]))

    def to_numpy(arr):
        return torch.squeeze(arr).numpy()

    length_finsler = to_numpy(dist["length_finsler"])
    length_riemann = to_numpy(dist["length_riemann"])
    xs = np.linspace(np.min(lengths), np.max(lengths))

    # compute KDE
    kde_lr = gaussian_kde(length_riemann, bw_method="scott")
    kde_lf = gaussian_kde(length_finsler, bw_method="scott")
    y_lr = kde_lr.pdf(xs)
    y_lf = kde_lf.pdf(xs)

    # compute mode, median and quartile
    moder = xs[np.argmax(y_lr)]
    modef = xs[np.argmax(y_lf)]
    middler, sdevr = np.mean(length_riemann), np.std(length_riemann)
    leftr, rightr = middler - sdevr, middler + sdevr
    middlef, sdevf = np.mean(length_finsler), np.std(length_finsler)
    leftf, rightf = middlef - sdevf, middlef + sdevf
    # leftr, middler, rightr = np.percentile(length_riemann, [25, 50, 75])
    # leftf, middlef, rightf = np.percentile(length_finsler, [25, 50, 75])

    # plot KDE distribution Finsler versus Riemann length functionals
    fig = plt.figure(0)
    ax0 = plt.axes()
    num_bins = 25
    # ax0.hist(length_riemann, num_bins, density=True, alpha=0.3, color='orange')
    # ax0.hist(length_finsler, num_bins, density=True, alpha=0.3, color='purple')
    ax0.plot(xs, y_lr, c="purple")
    ax0.fill_between(xs, y_lr, 0, facecolor="purple", alpha=0.1)
    ax0.vlines(moder, 0, kde_lr.pdf(moder), color="purple", ls=":")
    ax0.vlines(middler, 0, kde_lr.pdf(middler), color="purple", ls=":")
    ax0.fill_between(xs, 0, y_lr, where=(leftr <= xs) & (xs <= rightr), interpolate=True, facecolor="purple", alpha=0.2)

    ax0.plot(xs, y_lf, c="orange")
    ax0.fill_between(xs, y_lf, 0, facecolor="orange", alpha=0.1)
    ax0.vlines(modef, 0, kde_lf.pdf(modef), color="orange", ls=":")
    ax0.vlines(middlef, 0, kde_lf.pdf(middlef), color="orange", ls=":")
    ax0.fill_between(xs, 0, y_lf, where=(leftf <= xs) & (xs <= rightf), interpolate=True, facecolor="orange", alpha=0.2)

    ax0.set_xlabel("Lengths functionals")
    ax0.set_ylabel("Probability density")

    plt.show()
    fig.savefig(
        os.path.join(exp_folder, "stats{}_w_{}_samples.png".format(opts.title_model, len(length_finsler))), dpi=fig.dpi
    )


def plt_dist_diff(dist, ax1=None, xs=None, color="blue", label=None):
    def to_numpy(arr):
        return torch.squeeze(arr).numpy()

    length_finsler = to_numpy(dist["length_finsler"])
    length_riemann = to_numpy(dist["length_riemann"])
    length_diff = length_riemann - length_finsler
    # print(length_diff)
    plt.plot(length_diff)
    plt.show()
    if xs is None:
        xs = np.linspace(np.min(length_diff), np.max(length_diff))

    # KDE for probability distribution of the difference between Reimann and Finsler
    kde_diff = gaussian_kde(length_diff, bw_method="scott")
    y_diff = kde_diff.pdf(xs)
    mode_diff = xs[np.argmax(y_diff)]
    middle, sdev = np.mean(length_diff), np.std(length_diff)
    left, right = middle - sdev, middle + sdev

    # plot  distribution difference Finsler Riemann length functionals
    if ax1 is None:
        fig = plt.figure(1)
        ax1 = plt.axes()
    ax1.plot(xs, y_diff, c=color, label=label)
    ax1.fill_between(xs, y_diff, 0, facecolor=color, alpha=0.1)
    # ax1.vlines(mode_diff, 0, kde_diff.pdf(mode_diff), color=color, ls=':')
    ax1.vlines(middle, 0, kde_diff.pdf(middle), color=color, ls="-")
    # ax1.fill_between(xs, 0, y_diff, where=(left <= xs) & (xs <= right), interpolate=True, facecolor=color, alpha=0.2)
    ax1.set_xlabel(r"Difference of length functionals: $L_{G} - \mathcal{L}$")
    ax1.set_ylabel("Probability density")
    return ax1


def plt_dist_dim(distributions, labels, colors):
    fig = plt.figure(1)
    ax1 = plt.axes()
    # xs = np.linspace(-0.05, 0.1)
    xs = np.linspace(-0.2, 0.4)
    for i, dist in enumerate(distributions):
        ax1 = plt_dist_diff(dist, ax1, xs, colors[i], labels[i])
    ax1.legend()
    plt.show()


def investigation(opts, dist):
    gplvm_riemann, _, X = get_model(opts)
    length_finsler = dist["length_finsler"]
    length_riemann = dist["length_riemann"]
    spline_finsler = dist["spline_finsler"]
    spline_riemann = dist["spline_riemann"]
    length_neg = torch.squeeze((length_riemann - length_finsler) < 0)
    print(length_neg.shape)
    spline_finsler = spline_finsler[length_neg]
    spline_riemann = spline_riemann[length_neg]
    print(spline_riemann.shape)

    fig = plot_latent_space(opts, gplvm_riemann, X, spline_riemann, spline_finsler)


if __name__ == "__main__":
    opts = get_args()
    for opts.title_model in ["_3_400", "_5_400", "_10_400"]:
        print("options: {}".format(opts))
        exp_folder = os.path.join(opts.exp_folder, opts.data)

        # print("figures saved in:", exp_folder)
        # stats_geodesics(opts)

        dist = pickle_load(exp_folder, "distribution{}_w_{}.pkl".format(opts.title_model, opts.num_geod))
        investigation(opts, dist)
        plt_dist(dist)
        plt_dist_diff(dist)
        plt.show()

    # lentgh in latent space
    # dist_3 = pickle_load(exp_folder, "distribution_3_500_w_200.pkl")
    # dist_10 = pickle_load(exp_folder, "distribution_10_500_w_200.pkl")
    # dist_25 = pickle_load(exp_folder, "distribution_25_500_w_200.pkl")
    # dist_50 = pickle_load(exp_folder, "distribution_50_500_w_200.pkl")

    # length in obs space
    # dist_3 = pickle_load(exp_folder, "distribution_3_500_w_150.pkl")
    # dist_10 = pickle_load(exp_folder, "distribution_10_500_w_150.pkl")
    # dist_25 = pickle_load(exp_folder, "distribution_25_500_w_150.pkl")
    # dist_50 = pickle_load(exp_folder, "distribution_50_500_w_150.pkl")
    # distributions = [dist_3, dist_10, dist_25, dist_50]
    # colors = [ 'saddlebrown', 'peru', 'lightcoral', 'gold']
    # labels = ['dim 3', 'dim 10', 'dim 25', 'dim 50']
    # plt_dist_dim(distributions, labels, colors)
