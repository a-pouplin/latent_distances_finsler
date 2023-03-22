import networkx as nx
import numpy as np
import stochman
import torch
import torchplot as plt
from sklearn.decomposition import PCA
from stochman.discretized_manifold import DiscretizedManifold
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

if __name__ == "__main__":

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
    data_observ, label_tensor = all_data
    data_observ = data_observ.reshape(data_observ.shape[0], -1).to(torch.float)  # 5000x784
    num_data, dim_data = data_observ.shape

    # latent data
    cov = torch.cov(data_observ.t())
    values, vectors = torch.linalg.eigh(cov)
    proj = vectors[:, -2:] / values[-2:].sqrt().unsqueeze(0)
    data_latent = data_observ @ proj

    # Create metric
    model = stochman.manifold.LocalVarMetric(data=data_latent, sigma=0.1, rho=0.1)

    # with Discrete manifold
    with torch.no_grad():
        ran = torch.linspace(-1.0, 1.0, 10)
        gridX, gridY = torch.meshgrid([ran, ran], indexing="ij")
        grid = torch.stack((gridX.flatten(), gridY.flatten()), dim=1)  # 10000x2

    manifold = DiscretizedManifold()
    graph = manifold.fit(model=model, grid=[ran, ran], batch_size=100)

    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos=pos, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.show()

    raise
    plt.figure()
    # manifold_image = manifold.metric(grid).log().sum(dim=1).view(100, 100).t()
    # plt.imshow(manifold_image, extent=(ran[0], ran[-1], ran[0], ran[-1]), origin="lower")
    plt.plot(data_latent[::10, 0], data_latent[::10, 1], "w.", markersize=1)
    p0 = data_latent[torch.randint(high=num_data, size=[1], dtype=torch.long)]  # 5xD
    p1 = data_latent[torch.randint(high=num_data, size=[1], dtype=torch.long)]  # 5xD
    print("latent points:", p0.shape, p1.shape)
    C, success = manifold.connecting_geodesic(p0, p1)
    C.plot()
