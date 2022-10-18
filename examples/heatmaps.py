# latent space visualisations for a trained GPLVM model

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

from finsler.gplvm import gplvm
from finsler.utils.helper import create_filepath, create_folder, pickle_load
from finsler.visualisation.latent import volume_heatmap

matplotlib.rcParams["svg.fonttype"] = "none"


def get_args():
    parser = argparse.ArgumentParser()
    # data used
    parser.add_argument("--data", default="starfish_sphere", type=str)  # sphere or qPCR or font or starfish
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--model_folder", default="trained_models/", type=str)
    parser.add_argument("--exp_folder", default="plots/heatmaps/", type=str)
    parser.add_argument("--exp_num", default="9", type=str)  # number of the experiment to use for plotting
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))

    folderpath = os.path.abspath(str(opts.exp_folder) + "/" + str(opts.data) + "/" + str(opts.exp_num))
    create_folder(folderpath)
    print("--- figures saved in:", folderpath)

    # load previously training gplvm model
    model_folder = os.path.join(opts.model_folder, opts.data)
    model_saved = pickle_load(folder_path=model_folder, file_name="model_{}.pkl".format(opts.exp_num))
    model = model_saved["model"]
    Y = model_saved["Y"]
    X = model_saved["X"]
    lengthscale = model.base_model.kernel.lengthscale_unconstrained
    variance = model.base_model.kernel.variance_unconstrained
    print("Y shape: {} -- variance: {:.2f}, lengthscale: {:.2f}".format(Y.shape, variance, lengthscale))

    # get Riemannian and Finslerian metric
    gplvm_riemann = gplvm(model, mode="riemannian")
    gplvm_finsler = gplvm(model, mode="finslerian")
    X = model.X_loc.detach().numpy()

    # plot latent space with indicatrices
    # list_modes = ['variance', 'vol_riemann', 'vol_finsler', 'diff']
    list_modes = ["variance"]
    n_grid = 32
    for mode in list_modes:
        print("computing heatmaps... {}".format(mode))
        fig = plt.figure(0)
        ax = plt.axes()
        ax, im, hm_values, hm_ltt = volume_heatmap(ax, gplvm_riemann, X, mode, n_grid, vmin=-3)
        ax.scatter(X[:, 0], X[:, 1], marker="o", edgecolors="black", s=1)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        fig.colorbar(im)
        plt.title(str(mode))
        plt.show()

        filename = "res_" + str(n_grid) + "_" + str(mode) + ".png"
        filepath = create_filepath(folderpath, filename)
        fig.savefig(filepath, dpi=fig.dpi)
        print("--- heatmaps in latent space saved as: {}".format(filepath))
