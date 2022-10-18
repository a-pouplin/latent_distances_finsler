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


def make_torus_points(n_samples, noise=0, r=1, R=2):
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    phi = np.random.uniform(0, 2 * np.pi, n_samples)
    r = 1 - noise
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
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


def make_torus_surface(r=1, R=2):
    # Used only to plot the wireframe of the sphere for illustration
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, 2 * np.pi, 20)
    theta, phi = np.meshgrid(u, v)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


def on_sphere(x):
    # to check if the data points is on the sphere (as they should be)
    num_total = x.shape[0]
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
    num_onsphere = np.sum((r > 0.9) & (r < 1.1))
    return num_onsphere / num_total


def on_torus(points, r=1, R=2):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r_emp = np.sqrt((np.sqrt(x**2 + y**2) / R - 1) ** 2 + z**2)
    print(r_emp)
    error = (r_emp - r) / r_emp
    return np.sum(np.abs(error) < 0.1) / (points.shape[0])

    # to check if the data points is on the torus (as they should be)
    # num_total = x.shape[0]
    # r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
    # num_onsphere = np.sum((r > 0.9) & (r < 1.1))
    # return num_onsphere / num_total


def sample_vMF(mu, kappa, num_samples):
    """Generate num_samples N-dimensional samples from von Mises Fisher
    distribution around center mu \\in R^N with concentration kappa.
    """
    mu = np.array(mu) / np.linalg.norm(np.array(mu))
    dim = len(mu)
    result = np.zeros((num_samples, dim))
    for nn in range(num_samples):
        # sample offset from center (on sphere) with spread kappa
        w = _sample_weight(kappa, dim)
        # sample a point v on the unit sphere that's orthogonal to mu
        v = _sample_orthonormal_to(mu)
        # compute new point
        result[nn, :] = v * np.sqrt(1.0 - w**2) + w * mu

    return result


def _sample_weight(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4.0 * kappa**2 + dim**2) + 2 * kappa)
    x = (1.0 - b) / (1.0 + b)
    c = kappa * x + dim * np.log(1 - x**2)

    while True:
        z = np.random.beta(dim / 2.0, dim / 2.0)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u):
            return w


def _sample_orthonormal_to(mu):
    """Sample point on sphere orthogonal to mu."""
    v = np.random.randn(mu.shape[0])
    proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
    orthto = v - proj_mu_v
    return orthto / np.linalg.norm(orthto)


RADIAL_STD = 0.5
TGN_STD = 0.1
RATE = 0.05


def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    features = np.random.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.0
    labels = np.repeat(np.arange(num_classes), num_per_class)
    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))
    pinwheel = 10 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))
    return pinwheel / np.sqrt(np.linalg.norm(pinwheel))


def make_starfish(num_classes=5, num_per_class=100):
    v = make_pinwheel_data(RADIAL_STD, TGN_STD, num_classes, num_per_class, RATE)
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
    v = torch.tensor(make_pinwheel_data(RADIAL_STD, TGN_STD, num_classes, num_per_class, RATE))
    net = torch.nn.Sequential(
        torch.nn.Linear(2, int(dim / 2)), torch.nn.Sigmoid(), torch.nn.Linear(int(dim / 2), dim)
    ).double()
    y = net(v)
    y, v = y.detach().numpy(), v.detach().numpy()
    return y, v


def starfish_2sphere(num_classes=5, num_per_class=200):
    def projection_mercator(x, y):
        longitude = x
        latitude = 2 * np.arctan(np.exp(y)) - np.pi / 2
        obs_x = np.cos(latitude) * np.cos(longitude)
        obs_y = np.cos(latitude) * np.sin(longitude)
        obs_z = np.sin(latitude)
        return obs_x, obs_y, obs_z

    def projection_stereographic(x, y):
        norm = x**2 + y**2 + 1
        obs_x = 2 * x / norm
        obs_y = 2 * y / norm
        obs_z = (x**2 + y**2 - 1) / norm
        return obs_x, obs_y, obs_z

    latent_data = make_pinwheel_data(RADIAL_STD, TGN_STD, num_classes, num_per_class, RATE)
    obs_x, obs_y, obs_z = projection_stereographic(latent_data[:, 0], latent_data[:, 1])

    obs_data = np.stack([obs_x, obs_y, obs_z], axis=1)
    return obs_data, latent_data


# def highdim_starfish(dim = 10, num_classes = 5, num_per_class = 100):
#     # filling all the other dimensions with noise
#     v = make_pinwheel_data(RADIAL_STD, TGN_STD, num_classes, num_per_class, RATE)
#     v_noise = np.random.randn(v.shape[0], dim-2)
#     y = np.concatenate((v, v_noise), axis=1)
#     return y,v


def make_concentric_circles(num_per_circle=200, num_circles=2):
    points = np.empty((num_per_circle * num_circles, 2))
    radii = np.linspace(0, 1, num_circles + 1)[1:]
    for i, radius in enumerate(radii):
        thetas = np.random.uniform(-np.pi, np.pi, num_per_circle)
        points[i * (num_per_circle) : (i + 1) * num_per_circle, 0] = radius * np.cos(thetas)
        points[i * (num_per_circle) : (i + 1) * num_per_circle, 1] = radius * np.sin(thetas)
    return points


def highdim_circles(dim=3, num_per_circle=250, num_circles=2):
    torch.set_default_dtype(torch.float32)
    v = torch.tensor(make_concentric_circles(num_per_circle, num_circles))
    net = torch.nn.Sequential(
        torch.nn.Linear(2, int(dim / 2)), torch.nn.Sigmoid(), torch.nn.Linear(int(dim / 2), dim)
    ).double()
    y = net(v)
    y, v = y.detach().numpy(), v.detach().numpy()
    return y, v
