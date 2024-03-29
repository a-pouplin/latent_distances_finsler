import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from stochman.curves import CubicSpline
from stochman.discretized_manifold import DiscretizedManifold
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from finsler.gplvm import Gplvm
from finsler.kernels.rbf import RBF
from finsler.likelihoods.gaussian import Gaussian
from finsler.sasgp import SASGP
from finsler.utils.helper import create_filepath, pickle_load
from finsler.visualisation.latent import volume_heatmap


def get_args():
    parser = argparse.ArgumentParser()
    # manifold argments

    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--model_folder", default="models/sas/mnist", type=str)
    parser.add_argument("--model_title", default="mnist", type=str)
    parser.add_argument("--mode", default="riemannian", type=str)  # finslerian or riemannian
    parser.add_argument("--save_model", default=False, type=str)
    parser.add_argument("--num_geod", default=8, type=int)
    parser.add_argument("--res", default=10, type=int)  # resolution for the manifold grid
    parser.add_argument("--num_train", default=10000, type=int)
    parser.add_argument("--seed", default=12, type=int)
    parser.add_argument("--plot_images", default=False, type=bool)
    parser.add_argument("--plot_latent", default=True, type=bool)
    opts = parser.parse_args()
    return opts


def load_data(num_train):
    ## MNIST // TRAIN=60.000, TEST=10.000
    transform = transforms.ToTensor()
    trainset = MNIST(root="./data/", train=True, download=True, transform=transform)
    # testset = MNIST(root='./data/', train=False, download=True, transform=transform)

    # get a subset of the data
    trainset = torch.utils.data.Subset(trainset, range(num_train))
    # testset = torch.utils.data.Subset(trainset, range(500))

    # Create a data loader for the subset
    train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)

    # Concatenate the batches of data into a single tensor
    all_data = next(iter(train_loader))
    data_tensor, label_tensor = all_data
    return data_tensor, label_tensor


def load_sas_gplvm(model_folder, model_title):
    modelpath = os.path.join(model_folder, model_title + ".pt")
    model = pickle_load(folder_path=f"{model_folder}", file_name=f"{model_title}.pt")
    # times, loss = model['runtimes'], model['losses']
    model_params, parser_args = model["model"], model["args"]
    data_dimension = 784

    ## defining kernel and likelihood
    kernel_ls = model_params["kernel.length_scale"]
    kernel_a = model_params["kernel.variance"]
    sigma = torch.exp(model_params["likelihood.log_sigma"])
    kernel = RBF(
        length_scale=kernel_ls, variance=kernel_a, jitter=parser_args.jitter, input_dim=parser_args.latent_dim, ARD=True
    )
    likelihood = Gaussian(sigma=sigma, fit_noise=True)

    ## wrapping saved model into SASGP structure
    model = SASGP(
        kernel,
        likelihood,
        learning_rate=parser_args.lr,
        active_set=parser_args.num_active,
        latent_dim=parser_args.latent_dim,
        data_dim=data_dimension,
        data_size=parser_args.nof_observations,
    )
    model.load_state_dict(model_params)
    return model


def random_centered_points(center, radius, num_points):
    # generate random points inside a ball centered in {center} of radius {radius}
    # https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
    points = np.random.rand(num_points, len(center))
    points = points / np.linalg.norm(points, axis=1)[:, None]
    points = points * np.random.rand(num_points, 1) ** (1 / len(center))
    points = points * radius + center
    return torch.Tensor(points)


def initialise_sasmodel(model, data_tensor, label_tensor):
    model.y = torch.squeeze(data_tensor.data).reshape(-1, 784).float().detach()  # data in observed space (5000x784)
    num_data = model.y.shape[0]
    model.ylabels = label_tensor.detach()  # labels for data (5000,)
    model.X = model.amortization_net(model.y).detach()
    # model.Kinv = torch.linalg.solve(torch.eye(num_data), model.kernel.K(model.X, model.X)).detach()
    model.Kinv = torch.cholesky_inverse(
        model.kernel.K(model.X, model.X)
    ).detach()  # inverse of kernel matrix (5000x5000)
    return model


def plot_reconstruction(model, gplvm, num_imgs=10):
    # plot reconstruction of random images from the dataset
    img_list = np.random.randint(0, num_data, num_imgs)
    data_y = model.y[img_list]
    data_x = model.amortization_net(data_y).detach()
    reconst_y, _ = gplvm.embed(data_x.unsqueeze(0))
    reconst_y = reconst_y.squeeze(0).detach()

    # plot data_y and reconst_y
    plt.figure()
    for i in range(num_imgs):
        plt.subplot(2, num_imgs, i + 1)
        plt.imshow(data_y[i].reshape(28, 28))
        plt.axis("off")
        plt.subplot(2, num_imgs, num_imgs + i + 1)
        plt.imshow(reconst_y[i].reshape(28, 28))
        plt.axis("off")
    plt.show()


if __name__ == "__main__":

    opts = get_args()
    torch.manual_seed(opts.seed)

    modelpath_riemann = os.path.join(
        opts.model_folder, f"manifold_{opts.model_title}_res{opts.res}_with{opts.num_train}_riemannian.pkl"
    )
    modelpath_finsler = os.path.join(
        opts.model_folder, f"manifold_{opts.model_title}_res{opts.res}_with{opts.num_train}_finslerian.pkl"
    )

    print("We are loading the models:")
    print("Model path Riemannian: ", modelpath_riemann)
    print("Model path Finslerian: ", modelpath_finsler)

    # load data
    data_tensor, label_tensor = load_data(opts.num_train)

    # load the SAS-GPLVM model trained on fashion mnist
    sasmodel = load_sas_gplvm(opts.model_folder, opts.model_title)

    # add data to model:
    model = initialise_sasmodel(sasmodel, data_tensor, label_tensor)

    # for printing
    data_latent = model.X  # data in latent space (5000x2)
    num_data, _ = data_latent.shape  # 5000x2
    print("Data loaded, shape: ", model.y.shape, " and latent space shape: ", model.X.shape)

    # models wrapped with gplvm code to compute geodesics with stochman
    gplvm = Gplvm(model)  # Note that we only need the embed function from this class, so the mode is not used

    # dk, ddk = gplvm.evaluateDiffKernel(model.X[:5], model.X)

    # with Discrete manifold
    with torch.no_grad():
        ran = torch.linspace(-1.0, 1.0, opts.res)  # the higher the number of points, the more accurate the geodesics
        gridX, gridY = torch.meshgrid([ran, ran], indexing="ij")
        grid = torch.stack((gridX.flatten(), gridY.flatten()), dim=1)  # 100x2

    with open(modelpath_riemann, "rb") as file:
        manifold_riemann = pickle.load(file)
    assert isinstance(manifold_riemann, DiscretizedManifold), "Manifold should be of type DiscretizedManifold"

    with open(modelpath_finsler, "rb") as file:
        manifold_finsler = pickle.load(file)
    assert isinstance(manifold_finsler, DiscretizedManifold), "Manifold should be of type DiscretizedManifold"

    # start and end points for geodesics
    p0 = data_latent[torch.randint(high=num_data, size=[opts.num_geod], dtype=torch.long)]  # opts.num_geodxD
    p1 = data_latent[torch.randint(high=num_data, size=[opts.num_geod], dtype=torch.long)]  # opts.num_geodxD

    # define splines for both finsler and riemannian manifolds
    spline_riemann, _ = manifold_riemann.connecting_geodesic(p0, p1)
    spline_finsler, _ = manifold_finsler.connecting_geodesic(p0, p1)
    t = torch.linspace(0, 1, 100)
    curves_riemann = spline_riemann(t)
    curves_finsler = spline_finsler(t)

    # cubic spline from p0 to p1
    spline_euclidean = CubicSpline(p0, p1)
    lines = spline_euclidean(t)

    if opts.plot_images:
        # get mnist images on manifold
        num_images = 8
        y_img_riemann, _ = gplvm.embed(curves_riemann[:, :: int(len(t) / num_images), :])
        y_img_finsler, _ = gplvm.embed(curves_finsler[:, :: int(len(t) / num_images), :])
        y_img_euclidean, _ = gplvm.embed(lines[:, :: int(len(t) / num_images), :])

        y_img_riemann = y_img_riemann.reshape(opts.num_geod, -1, 28, 28).detach().numpy()
        y_img_finsler = y_img_finsler.reshape(opts.num_geod, -1, 28, 28).detach().numpy()
        y_img_euclidean = y_img_euclidean.reshape(opts.num_geod, -1, 28, 28).detach().numpy()

        # plot images along the euclidean, riemannian and finslerian curve

        # figure with fashion mnsit images
        fig, axs = plt.subplots(opts.num_geod, num_images, figsize=(num_images, opts.num_geod))
        for i in range(opts.num_geod):
            for j in range(num_images):
                axs[i, j].imshow(y_img_euclidean[i, j, :, :], cmap="gray")
                axs[i, j].axis("off")
        fig.text(0.5, 0.04, "Euclidean", ha="center")
        fig.savefig(opts.model_folder + "/images_mnist_euclidean.svg")

        # figure with fashion mnsit images riemannian
        fig2, axs2 = plt.subplots(opts.num_geod, num_images, figsize=(num_images, opts.num_geod))
        for i in range(opts.num_geod):
            for j in range(num_images):
                axs2[i, j].imshow(y_img_riemann[i, j, :, :], cmap="gray")
                axs2[i, j].axis("off")
        fig2.text(0.5, 0.04, "Riemannian", ha="center")
        fig2.savefig(opts.model_folder + "/images_mnist_riemann.svg")

        # figure with fashion mnsit images finslerian
        fig3, axs3 = plt.subplots(opts.num_geod, num_images, figsize=(num_images, opts.num_geod))
        for i in range(opts.num_geod):
            for j in range(num_images):
                axs3[i, j].imshow(y_img_finsler[i, j, :, :], cmap="gray")
                axs3[i, j].axis("off")
        fig3.text(0.5, 0.04, "Finslerian", ha="center")
        fig3.savefig(opts.model_folder + "/images_mnist_finsler.svg")

    # plot with images of mnist
    if opts.plot_latent:
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        mnist_y = model.y[::10].detach().numpy().reshape((-1, 28, 28))
        mnist_x = data_latent[::10]
        mnist_label = label_tensor[::10]

        fig4 = plt.figure(1, figsize=(10, 10))
        ax4 = plt.axes()
        # ax4, heatmap, _, _ = volume_heatmap(ax4, gplvm, data_latent, mode="variance", n_grid=15)
        for n, image in enumerate(mnist_y):
            im = OffsetImage(image, zoom=0.5, cmap=plt.cm.gray, alpha=0.8)
            ab = AnnotationBbox(im, (mnist_x[n, 0], mnist_x[n, 1]), xycoords="data", frameon=False)
            ax4.add_artist(ab)
        ax4.scatter(mnist_x[:, 0], mnist_x[:, 1], c="k", s=1)
        spline_finsler.plot(color="orange", linewidth=2.0, zorder=1e6, label="Finsler geodesic")
        spline_riemann.plot(color="purple", linewidth=2.0, zorder=2e6, label="Riemann geodesic", linestyle=(0, (5, 10)))
        spline_euclidean.plot(
            color="gray", linewidth=2.0, zorder=1e6, label="Euclidean geodesic", linestyle="--", alpha=0.5
        )
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        ax4.legend()
        ax4.set_aspect("equal")
        ax4.axis("off")

        # cax = fig4.add_axes([ax4.get_position().x1 + 0.01, ax4.get_position().y0, 0.02, ax4.get_position().height])
        # fig4.colorbar(heatmap, cax=cax)
        fig4.savefig(opts.model_folder + "/latent_mnist_withimages_{}.svg".format(opts.mode))
        plt.show()
        print(
            "--- plot of the latent space with images saved as: {}".format(
                opts.model_folder + "/latent_mnist_images_{}.svg".format(opts.mode)
            )
        )

        mnist_y = model.y[::1].detach().numpy().reshape((-1, 28, 28))
        mnist_x = data_latent[::1]
        mnist_label = label_tensor[::1]
        fig5 = plt.figure(1, figsize=(10, 10))
        ax5 = plt.axes()
        ax5, heatmap, _, _ = volume_heatmap(ax5, gplvm, data_latent, mode="variance", n_grid=25, vmin=0.54)
        # print color labels
        ax5.scatter(mnist_x[:, 0], mnist_x[:, 1], c=mnist_label, s=2, alpha=0.8, cmap="tab10")
        spline_finsler.plot(color="orange", linewidth=2.0, zorder=1e6, label="Finsler geodesic")
        spline_riemann.plot(color="purple", linewidth=2.0, zorder=2e6, label="Riemann geodesic", linestyle=(0, (5, 10)))
        spline_euclidean.plot(
            color="gray", linewidth=2.0, zorder=1e6, label="Euclidean geodesic", linestyle="--", alpha=0.5
        )
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        ax5.set_aspect("equal")
        cax = fig5.add_axes([ax5.get_position().x1 + 0.01, ax5.get_position().y0, 0.02, ax5.get_position().height])
        fig5.colorbar(heatmap, cax=cax, alpha=0.5)
        fig5.savefig(opts.model_folder + "/latent_mnist_withoutimages_{}.svg".format(opts.mode))
        ax5.legend()
        ax5.axis("off")
        plt.show()
        print(
            "--- plot of the latent space with images saved as: {}".format(
                opts.model_folder + "/latent_mnist_images_{}.svg".format(opts.mode)
            )
        )
