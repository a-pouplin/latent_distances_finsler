import numpy as np
import torch
from scipy.spatial.transform import Rotation


def make_sphere_points(n_samples, noise=0):
    # noise in percentage, sphere with radius r=1
    np.random.seed(seed=42)
    theta = np.random.uniform(0, np.pi, n_samples)
    phi = np.random.uniform(0, 2 * np.pi, n_samples)
    r = 1 - noise
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))


def sample_sphere(npoints):
    vec = np.random.randn(npoints, 3)
    vec = vec / np.linalg.norm(vec, axis=1)[:, None]
    return vec


def remove_points(points, center, radius):
    center = np.array(center) / np.linalg.norm(np.array(center))
    recenter_points = points - center
    recenter_norm = np.linalg.norm(recenter_points, axis=1)
    return points[recenter_norm > radius]


def make_sphere_surface(r=1):
    # Used only to plot the wireframe of the sphere for illustration
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def on_sphere(x, error=0.1):
    # to check if the data points is on the sphere (as they should be)
    num_total = x.shape[0]
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
    num_onsphere = np.sum((r > 1 - error) & (r < 1 + error))
    return num_onsphere / num_total


# 0.5, 0.1, 0.1
def make_pinwheel_data(num_classes, num_per_class, radial_std=0.5, tangential_std=0.1, rate=0.07):
    np.random.seed(seed=42)
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    features = np.random.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.0
    labels = np.repeat(np.arange(num_classes), num_per_class)
    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))
    pinwheel = 10 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))
    return pinwheel / np.sqrt(np.linalg.norm(pinwheel))


def make_pinwheel_data_alacilie(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.0
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))
    v = 10 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))
    return v / np.sqrt(np.linalg.norm(v))


def make_starfish(num_classes=5, num_per_class=100):
    v = make_pinwheel_data(num_classes, num_per_class)
    z_noise = np.random.normal(0, 0.01, size=[num_classes * num_per_class])
    xs = (v[:, 0] - np.mean(v[:, 0])) / np.std(v[:, 0])
    ys = (v[:, 1] - np.mean(v[:, 1])) / np.std(v[:, 1])
    vec = np.empty([num_classes * num_per_class, 3])
    vec[:, 0] = xs
    vec[:, 1] = ys
    vec[:, 2] = z_noise
    r_x = Rotation.from_euler("x", 45, degrees=True)
    r_y = Rotation.from_euler("y", 20, degrees=True)
    r_z = Rotation.from_euler("z", 75, degrees=True)
    p = r_z.apply(r_y.apply(r_x.apply(vec)))
    return p, v
    # return torch.tensor(p, dtype=torch.float32)


def highdim_starfish(dim=10, num_classes=5, num_per_class=100):
    torch.set_default_dtype(torch.float32)
    v = torch.tensor(make_pinwheel_data(num_classes, num_per_class))
    net = torch.nn.Sequential(
        torch.nn.Linear(2, int(dim / 2)), torch.nn.Sigmoid(), torch.nn.Linear(int(dim / 2), dim)
    ).double()
    y = net(v)
    y, v = y.detach().numpy(), v.detach().numpy()
    return y, v


def projection_stereographic(x, y):
    norm = x**2 + y**2 + 1
    obs_x = 2 * x / norm
    obs_y = 2 * y / norm
    obs_z = (x**2 + y**2 - 1) / norm
    return obs_x, obs_y, obs_z


def starfish_2sphere(num_classes=5, num_per_class=200):

    latent_data = make_pinwheel_data(num_classes, num_per_class)
    obs_x, obs_y, obs_z = projection_stereographic(latent_data[:, 0], latent_data[:, 1])

    obs_data = np.stack([obs_x, obs_y, obs_z], axis=1)
    return obs_data, latent_data


def make_cheese_data(num_holes, radius_holes, num_data, center_holes):
    # create data points uniformely distributed and remove data to shape holes in the latent space
    np.random.seed(seed=42)
    data = np.random.uniform(-1, 1, size=[num_data, 2])
    if center_holes is None:
        center_holes = np.random.uniform(-1 + radius_holes, 1 - radius_holes, size=[num_holes, 2])
    for i in range(num_holes):
        data = data[np.linalg.norm(data - center_holes[i], axis=1) > radius_holes]
    return data


def cheese_2sphere(num_holes=3, radius_holes=0.2, num_data=4000, center_holes=None):

    latent_data = make_cheese_data(num_holes, radius_holes, num_data, center_holes)
    obs_x, obs_y, obs_z = projection_stereographic(latent_data[:, 0], latent_data[:, 1])

    obs_data = np.stack([obs_x, obs_y, obs_z], axis=1)
    return obs_data, latent_data


def make_chessboard_data(num_grids, num_points):
    # Compute the grid size and the number of points per grid
    grid_size = 1.0 / num_grids
    points_per_grid = int(num_points / (num_grids * num_grids))

    # Generate the grid coordinates
    grid_coords = np.linspace(0, 1, num_grids + 1)

    # Generate the points in a chessboard pattern
    points = []
    for i in range(num_grids):
        for j in range(num_grids):
            if (i + j) % 2 == 0:
                x_coords = np.random.uniform(grid_coords[i], grid_coords[i + 1], points_per_grid)
                y_coords = np.random.uniform(grid_coords[j], grid_coords[j + 1], points_per_grid)
                points.extend(list(zip(x_coords, y_coords)))
    # Convert the points to a numpy array and return it
    # centered at the origin and scaled to [-1, 1]
    points = 1.5 * (np.array(points) - 0.5)
    return points


def cheesboard_2sphere(num_grids=5, num_points=2000):
    latent_data = make_chessboard_data(num_grids, num_points)
    obs_x, obs_y, obs_z = projection_stereographic(latent_data[:, 0], latent_data[:, 1])

    obs_data = np.stack([obs_x, obs_y, obs_z], axis=1)
    return obs_data, latent_data


def highdim_starfish_alacilie(num_classes=5, num_per_class=100):
    v = torch.tensor(make_pinwheel_data_alacilie(0.75, 0.15, num_classes, num_per_class, 0.025))
    v = v.to(torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 16))
    y = net(v)
    return y, v


def make_concentric_circle_data(num_classes, num_per_class):
    np.random.seed(seed=42)
    data = []
    for r in range(1, num_classes + 1):
        theta = np.random.uniform(0, 2 * np.pi, num_per_class)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        data.extend(list(zip(x, y)))
    return np.array(data) / (num_classes + 1)


def concentric_2sphere(num_classes=5, num_per_class=100):
    latent_data = make_concentric_circle_data(num_classes, num_per_class)
    obs_x, obs_y, obs_z = projection_stereographic(latent_data[:, 0], latent_data[:, 1])

    obs_data = np.stack([obs_x, obs_y, obs_z], axis=1)
    return obs_data, latent_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # # v = make_pinwheel_data_alacilie(0.75, 0.15, 5, 50, 0.025)
    # v = make_concentric_circle_data(5, 100)
    # # x = make_chessboard_data()
    # plt.scatter(v[:, 0], v[:, 1], color='blue')
    # plt.show()
    # raise
    # X = make_pinwheel_data(num_classes=5, num_per_class=200, radial_std=0.5, tangential_std=0.1, rate=0.06)
    # Y, X = cheese_2sphere(num_holes=1, radius_holes=0.4, num_data=4000, center_holes=np.zeros(2))
    Y, X = concentric_2sphere()
    # plt.scatter(X[:, 0], X[:, 1])
    fig = plt.figure(2)
    ax = plt.axes(projection="3d")
    ax.set_box_aspect([1, 1, 1])
    ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2])
    ax.set_xlim((-1, 1)), ax.set_ylim((-1, 1)), ax.set_zlim((-1, 1))
    plt.show()
