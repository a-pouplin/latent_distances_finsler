import argparse
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import stochman
import torch
import torchvision
from stochman.curves import CubicSpline
from stochman.discretized_manifold import DiscretizedManifold
from stochman.geodesic import geodesic_minimizing_energy
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, ImageFolder

from finsler.gplvm import Gplvm
from finsler.kernels.rbf import RBF
from finsler.likelihoods.gaussian import Gaussian
from finsler.sasgp import SASGP
from finsler.utils.helper import create_filepath, pickle_load


def get_args():
    parser = argparse.ArgumentParser()
    # manifold argments

    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--model_folder", default="models/sas/fmnist", type=str)
    parser.add_argument("--model_title", default="fmnist", type=str)
    parser.add_argument("--mode", default="riemannian", type=str)  # finslerian or riemannian
    parser.add_argument("--save_model", default=False, type=str)
    parser.add_argument("--num_geod", default=5, type=int)
    parser.add_argument("--res", default=32, type=int)  # resolution for the manifold grid
    parser.add_argument("--num_train", default=5000, type=int)
    opts = parser.parse_args()
    return opts


def load_data(num_train):
    ## MNIST // TRAIN=60.000, TEST=10.000
    transform = transforms.ToTensor()
    trainset = FashionMNIST(root="./data/", train=True, download=True, transform=transform)
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


def load_model(model_folder, model_title):
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


if __name__ == "__main__":

    opts = get_args()
    modelpath = os.path.join(
        opts.model_folder, f"manifold_{opts.model_title}_res{opts.res}_with{opts.num_train}_{opts.mode}.pkl"
    )
    print("everything will be saved in:", modelpath)

    # load data
    data_tensor, label_tensor = load_data(opts.num_train)

    # load model
    model = load_model(opts.model_folder, opts.model_title)

    # add data to model:
    model.y = torch.squeeze(data_tensor.data).reshape(-1, 784).float().detach()  # data in observed space (5000x784)
    model.ylabels = label_tensor.detach()  # labels for data (5000,)
    model.X = model.amortization_net(model.y).detach()
    data_latent = model.X  # data in latent space (5000x2)
    model.Kinv = torch.cholesky_inverse(
        model.kernel.K(data_latent, data_latent)
    ).detach()  # inverse of kernel matrix (5000x5000)

    # for printing
    num_data, _ = data_latent.shape  # 5000x2
    print("Data loaded, shape: ", model.y.shape, " and latent space shape: ", model.X.shape)

    # models wrapped with gplvm code to compute geodesics with stochman
    gplvm = Gplvm(model, mode=opts.mode)

    # with Discrete manifold
    with torch.no_grad():
        ran = torch.linspace(-1.0, 1.0, opts.res)  # the higher the number of points, the more accurate the geodesics
        gridX, gridY = torch.meshgrid([ran, ran], indexing="ij")
        grid = torch.stack((gridX.flatten(), gridY.flatten()), dim=1)  # 100x2

    if opts.save_model:
        manifold = DiscretizedManifold()
        manifold.fit(model=gplvm, grid=[ran, ran], batch_size=256)
        print("Manifold fitted. Saving model.....")
        # save manifold
        with open(modelpath, "wb") as f:
            pickle.dump(manifold, f, pickle.HIGHEST_PROTOCOL)
        print("model saved !")

    with open(modelpath, "rb") as file:
        manifold = pickle.load(file)
    assert isinstance(manifold, DiscretizedManifold), "Manifold should be of type DiscretizedManifold"

    # start and end points for geodesics
    torch.manual_seed(2)  # fix seed for reproducibility
    # p0 = data_latent[torch.randint(high=num_data, size=[opts.num_geod], dtype=torch.long)]  # opts.num_geodxD
    # p1 = data_latent[torch.randint(high=num_data, size=[opts.num_geod], dtype=torch.long)]  # opts.num_geodxD
    p0 = random_centered_points(center=[-0.5, 0.0], radius=0.2, num_points=opts.num_geod)
    p1 = random_centered_points(center=[0.5, -0.3], radius=0.2, num_points=opts.num_geod)

    spline_manifold, _ = manifold.connecting_geodesic(p0, p1)
    t = torch.linspace(0, 1, 100)
    curves = spline_manifold(t)
    # cubic spline from p0 to p1
    spline_euclidean = CubicSpline(p0, p1)
    lines = spline_euclidean(t)

    # plot geodesics and euclidean distances
    # plt.figure()
    # for i in range(opts.num_geod):
    #     plt.plot(curves[i, :,  0].detach().numpy(), curves[i, :, 1].detach().numpy(), 'k')
    #     plt.plot(lines[i, :,  0].detach().numpy(), lines[i, :, 1].detach().numpy(), 'r')
    # plt.show()

    # get mnist images on manifold
    num_images = 20
    y_img_manifold, _ = gplvm.embed(curves[:, :: int(len(t) / num_images), :])
    y_img_manifold = y_img_manifold.reshape(opts.num_geod, -1, 28, 28).detach().numpy()
    # get mnist images in euclidean space
    y_img_euclidean, _ = gplvm.embed(lines[:, :: int(len(t) / num_images), :])
    y_img_euclidean = y_img_euclidean.reshape(opts.num_geod, -1, 28, 28).detach().numpy()

    # plot images of mnist obtained along an euclidean and "manifold" curve
    fig1, axs1 = plt.subplots(opts.num_geod, num_images, figsize=(num_images, opts.num_geod))
    for i in range(opts.num_geod):
        for j in range(num_images):
            axs1[i, j].imshow(y_img_manifold[i, j], cmap="Greys", interpolation="nearest")
            axs1[i, j].axis("off")
    plt.title("Geodesics on a {} manifold".format(opts.mode))
    fig1.savefig(opts.model_folder + "/images_fmnist_{}.png".format(opts.mode))

    fig2, axs2 = plt.subplots(opts.num_geod, num_images, figsize=(num_images, opts.num_geod))
    for i in range(opts.num_geod):
        for j in range(num_images):
            axs2[i, j].imshow(y_img_euclidean[i, j], cmap="Greys", interpolation="nearest")
            axs2[i, j].axis("off")
    plt.title("Geodesics on an euclidean manifold")
    fig2.savefig(opts.model_folder + "/images_fmnist_euclidean.png")

    # # plot manifold and geodesics in latent space
    # fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))
    # axs3.scatter(data_latent[::10, 0], data_latent[::10, 1], c=label_tensor[::10], s=2, alpha=0.8, cmap='tab10')
    # sm = plt.cm.ScalarMappable(cmap='tab10')
    # fig3.colorbar(sm)
    # spline_manifold.plot(color='k', linewidth=1)
    # spline_euclidean.plot(color='r', linewidth=1)
    # plt.title("Geodesic in the latent space")
    # filename = "latent_fmnist_{}.png".format(opts.mode)
    # filepath = os.path.join(opts.model_folder, filename)
    # fig3.savefig(filepath)
    # print("--- plot of the latent space saved as: {}".format(filepath))

    # plot with images of mnist
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    mnist_y = model.y[::10].detach().numpy().reshape((-1, 28, 28))
    mnist_x = data_latent[::10]
    fig4, axs4 = plt.subplots(1, 1, figsize=(5, 5))
    for n, image in enumerate(mnist_y):
        im = OffsetImage(image, zoom=0.3, cmap=plt.cm.gray, alpha=0.5)
        ab = AnnotationBbox(im, (mnist_x[n, 0], mnist_x[n, 1]), xycoords="data", frameon=False)
        axs4.add_artist(ab)
    spline_manifold.plot(color="orange", linewidth=1, zorder=-1)
    spline_euclidean.plot(color="k", linewidth=1, zorder=-1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
    filename = "latent_fmnist_images_{}.png".format(opts.mode)
    filepath = os.path.join(opts.model_folder, filename)
    plt.savefig(filepath)
    print("--- plot of the latent space with images saved as: {}".format(filepath))
