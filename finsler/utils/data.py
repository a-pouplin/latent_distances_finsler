import numpy as np

def make_sphere_points(n_samples, noise=0):
    # noise in percentage, sphere with radius r=1
    np.random.seed(seed=42)
    theta = np.random.uniform(0, np.pi, n_samples)
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    r = 1-noise
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.column_stack((x,y,z))

def sample_sphere(npoints):
    vec = np.random.randn(npoints, 3)
    vec = vec/np.linalg.norm(vec, axis=1)[:,None]
    return vec

def remove_points(points, center, radius):
    center = np.array(center)/np.linalg.norm(np.array(center))
    recenter_points = points - center
    recenter_norm = np.linalg.norm(recenter_points, axis=1)
    return points[recenter_norm>radius]

def make_sphere_surface(n_samples):
    # Used only to plot the wireframe of the sphere for illustration
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    return x,y,z

def on_sphere(x):
    # to check if the data points is on the sphere (as they should be)
    num_total = x.shape[0]
    r = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
    num_onsphere = np.sum((r> 0.95) & (r<1.05))
    return r, num_onsphere/num_total


def sample_vMF(mu, kappa, num_samples):
    """Generate num_samples N-dimensional samples from von Mises Fisher
    distribution around center mu \in R^N with concentration kappa.
    """
    mu = np.array(mu)/np.linalg.norm(np.array(mu))
    dim = len(mu)
    result = np.zeros((num_samples, dim))
    for nn in range(num_samples):
        # sample offset from center (on sphere) with spread kappa
        w = _sample_weight(kappa, dim)
        # sample a point v on the unit sphere that's orthogonal to mu
        v = _sample_orthonormal_to(mu)
        # compute new point
        result[nn, :] = v * np.sqrt(1. - w ** 2) + w * mu

    return result

def _sample_weight(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)
    x = (1. - b) / (1. + b)
    c = kappa * x + dim * np.log(1 - x ** 2)

    while True:
        z = np.random.beta(dim / 2., dim / 2.)
        w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
            return w


def _sample_orthonormal_to(mu):
    """Sample point on sphere orthogonal to mu."""
    v = np.random.randn(mu.shape[0])
    proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
    orthto = v - proj_mu_v
    return orthto / np.linalg.norm(orthto)

