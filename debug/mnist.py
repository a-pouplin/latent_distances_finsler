import argparse
import os

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

from finsler.distributions import NonCentralNakagami
from finsler.gplvm import Gplvm
from finsler.kernels.rbf import RBF
from finsler.likelihoods.gaussian import Gaussian
from finsler.sasgp import SASGP
from finsler.utils.helper import pickle_load


def get_args():
    parser = argparse.ArgumentParser()
    # manifold argments
    parser.add_argument("--num_geod", default=3, type=int)  # num of random geodesics to plot
    parser.add_argument("--iter_energy", default=100, type=int)  # num of steps to minimise energy func
    # data used
    parser.add_argument("--data", default="mnist", type=str)  # sphere or starfish or vMF
    # load previous exp
    parser.add_argument("--train", action="store_false")
    parser.add_argument("--exp_folder", default="plots/sas/", type=str)
    parser.add_argument("--model_folder", default="models/sas/", type=str)
    parser.add_argument("--model_title", default="mnist", type=str)
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    # load data

    ## MNIST // TRAIN=60.000, TEST=10.000
    transform = transforms.ToTensor()
    trainset = MNIST(root="./data/", train=True, download=True, transform=transform)
    testset = MNIST(root="./data/", train=False, download=True, transform=transform)

    # Data Loaders / Train & Test
    # get a subset of the data
    trainset = torch.utils.data.Subset(trainset, range(5000))
    testset = torch.utils.data.Subset(trainset, range(500))

    # Create a data loader for the subset
    train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)

    # Concatenate the batches of data into a single tensor
    all_data = next(iter(train_loader))
    data_tensor, label_tensor = all_data

    # load model
    opts = get_args()
    modelpath = os.path.join(opts.model_folder, opts.model_title + ".pt")
    model = pickle_load(folder_path=f"{opts.model_folder}", file_name=f"{opts.model_title}.pt")
    times, loss, model_params, parser_args = model["runtimes"], model["losses"], model["model"], model["args"]
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

    # add data to model:
    model.y = torch.squeeze(data_tensor.data).reshape(-1, 784).float().detach()
    model.ylabels = label_tensor.detach()
    model.X = model.amortization_net(model.y).detach()
    data = model.X
    num_data, dim_data = data.shape
    model.Kinv = torch.cholesky_inverse(model.kernel.K(data, data)).detach()

    print("Data loaded, shape: ", model.y.shape)
    # plot latent space with labels
    # ax = plt.figure(figsize=(10, 10))
    # plt.scatter(model.X[::100, 0].detach().numpy(), model.X[::100, 1].detach().numpy(), c=mnist_train.dataset.train_labels[::100], cmap='tab10')
    # plt.colorbar()
    # plt.show()

    # xstar
    num_points = 50
    xstar = torch.empty((num_points, 2))
    xstar[:, 0], xstar[:, 1] = torch.linspace(0, 1, num_points), torch.linspace(0, 1, num_points)

    gplvm_riemann = Gplvm(model, mode="riemannian")
    gplvm_finsler = Gplvm(model, mode="finslerian")
    mu, var = gplvm_finsler.embed(xstar)

    # Energy function computed with riemannian metric
    optimizer = torch.optim.LBFGS
    eval_grid = 20
    dim_latent, dim_obs = 2, 784

    c_coords_riemann, c_obs_riemann = torch.empty((opts.num_geod, eval_grid, dim_latent)), np.empty(
        (opts.num_geod, eval_grid, dim_obs)
    )
    c_coords_finsler, c_obs_finsler = torch.empty((opts.num_geod, eval_grid, dim_latent)), np.empty(
        (opts.num_geod, eval_grid, dim_obs)
    )
    t = torch.linspace(0, 1, eval_grid)

    # # computing geodesic curves along the starfish branches
    # x1 = [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5]]
    # x0 = np.tile([0.5, 0.5], (len(x1), 1))

    # for i in range(opts.num_geod):
    #     start, ends = torch.tensor([x0[i]], dtype=torch.float), torch.tensor([x1[i]], dtype=torch.float)

    #     print("\n")
    #     print("Riemannian")
    #     curve_r = CubicSpline(start, ends, requires_grad=True)
    #     geodesic_minimizing_energy(curve_r, gplvm_riemann, optimizer, opts.iter_energy, eval_grid)

    #     # print("\n")
    #     # print("Finslerian")
    #     # curve_f = CubicSpline(start, ends, requires_grad=True)
    #     # geodesic_minimizing_energy(curve_f, gplvm_finsler, optimizer, opts.iter_energy, eval_grid)

    #     with torch.no_grad():
    #         c_coords_riemann[i, :, :] = curve_r(t)
    #         c_obs_riemann[i, :, :], _ = gplvm_riemann.embed(c_coords_riemann[i, :, :].squeeze())

    #         # c_coords_finsler[i, :, :] = curve_f(t)
    #         # c_obs_finsler[i, :, :], _ = gplvm_finsler.embed(c_coords_finsler[i, :, :].squeeze())

    # # with Latent Oddity
    # riemann_metric = stochman.manifold.LocalVarMetric(data=data, sigma=0.1, rho=0.1)
    # plt.scatter(data[::10, 0].numpy(), data[::10, 1].numpy(), c=model.ylabels[::10], s=2, cmap='tab10')
    # plt.colorbar()
    # p0 = data[torch.randint(high=num_data, size=[10], dtype=torch.long)]  # 10xD
    # p1 = data[torch.randint(high=num_data, size=[10], dtype=torch.long)]  # 10xD
    # C, success = riemann_metric.connecting_geodesic(p0, p1)
    # C.plot()

    # x =torch.as_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
    #         3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
    #         5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8,
    #         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9])
    # y =torch.as_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3,
    #         4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7,
    #         8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
    #         2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # xn =torch.as_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
    #         2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
    #         4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
    #         7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])
    # yn =torch.as_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3,
    #         4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7,
    #         8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
    #         2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    model = gplvm_finsler
    # grid = [torch.linspace(-1., 1., 10),torch.linspace(-1., 1., 10)]
    t, bs = torch.linspace(0, 1, 50), 10
    line = CubicSpline(begin=torch.zeros(bs, 2), end=torch.ones(bs, 2))
    # line.begin = torch.cat([grid[0][x].view(-1, 1), grid[1][y].view(-1, 1)], dim=1)  # (bs)x2
    # line.end = torch.cat([grid[0][xn].view(-1, 1), grid[1][yn].view(-1, 1)], dim=1)  # (bs)x2
    # weights = model.curve_length(line(t))

    coords = line(t)  # (bs, num_data, dim_data)
    loc_der, cov_der = model.compute_derivatives_batch(coords)  # (bs, num_data - 1, dim_data), (bs, num_data - 1)
    nck = NonCentralNakagami(loc_der, cov_der)

    print(nck.expectation().shape)
    print(nck.variance().shape)

    energy = (nck.expectation() ** 2).sum(dim=-1)
    # with Discrete manifold
    with torch.no_grad():
        ran = torch.linspace(-1.0, 1.0, 10)
        gridX, gridY = torch.meshgrid([ran, ran], indexing="ij")
        grid = torch.stack((gridX.flatten(), gridY.flatten()), dim=1)  # 100x2

    manifold = DiscretizedManifold()
    manifold.fit(model=model, grid=[ran, ran], batch_size=100)

    ## plot networks
    # pos = nx.spring_layout(manifold.G)
    # nx.draw_networkx(manifold.G, pos=pos, with_labels=True, node_color='lightblue', edge_color='gray')
    # plt.show()

    # raise
    p0 = data[torch.randint(high=num_data, size=[5], dtype=torch.long)]  # 5xD
    p1 = data[torch.randint(high=num_data, size=[5], dtype=torch.long)]  # 5xD
    print("latent points:", p0.shape, p1.shape)
    C, success = manifold.connecting_geodesic(p0, p1)

    plt.figure()
    plt.scatter(data[::10, 0], data[::10, 1], c=label_tensor[::10], s=2, alpha=0.8, cmap="tab10")
    C.plot()
    plt.colorbar()
    plt.title("Geodesic in the latent space")
    plt.show()
