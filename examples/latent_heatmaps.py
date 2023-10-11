# latent space visualisations for a trained GPLVM model

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

from finsler.gplvm import Gplvm
from finsler.utils.helper import create_filepath, create_folder, pickle_load
from finsler.visualisation.latent import volume_heatmap

matplotlib.rcParams["svg.fonttype"] = "none"


def get_args():
    parser = argparse.ArgumentParser()
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--model_folder", default="models/starfish/", type=str)
    parser.add_argument("--exp_folder", default="plots/latent_heatmaps/", type=str)
    parser.add_argument("--model_title", default="model_2", type=str)  # number of the experiment to use for plotting
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":

    opts = get_args()
    print("options: {}".format(opts))

    folderpath = os.path.abspath(str(opts.exp_folder) + "/" + str(opts.model_title))
    create_folder(folderpath)
    print("--- figures saved in:", folderpath)

    # load previously training gplvm model
    model = pickle_load(folder_path=f"{opts.model_folder}", file_name=f"{opts.model_title}.pkl")
    Y = model.y.data.numpy().transpose()
    X = model.X.data.numpy()
    lengthscale = model.kernel.lengthscale_unconstrained.data
    variance = model.kernel.variance_unconstrained.data

    # get Riemannian and Finslerian metric
    gplvm_riemann = Gplvm(model, mode="riemannian")
    gplvm_finsler = Gplvm(model, mode="finslerian")

    # plot latent space with indicatrices
    list_modes = ["variance", "vol_riemann", "vol_finsler", "diff"]
    n_grid = 8
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
