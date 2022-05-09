import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path
from scipy.interpolate import griddata


def pickle_save(data, folder_path, file_name):
    data_path = folder_path
    if not os.path.exists(data_path):
        print("creating folder for experiments: {}".format(data_path))
        os.mkdir(data_path)
    with open(os.path.join(data_path, file_name), "wb") as fw:
        pickle.dump(data, fw)


def pickle_load(folder_path, file_name):
    data_path = os.path.join(folder_path, file_name)
    with open(data_path, "rb") as fr:
        data = pickle.load(fr)
    return data


def indicatrix_finsler(mu, var, point):
    """NOT USED. Should use func: 'indicatrices_finsler' instead"""
    from geoml.stats import NonCentralNakagami

    mu, var = torch.squeeze(mu), torch.squeeze(var)
    vectors = torch.linspace(-2, 2, 32)
    size = len(vectors)
    finsler = torch.empty((size, size))
    for i1, y1 in enumerate(vectors):
        for i2, y2 in enumerate(vectors):
            y = torch.unsqueeze(torch.tensor([y1, y2]), 0)  # random vectors
            var_x = y.mm(var).mm(y.T)
            mu_x = y.mm(mu).mm(mu.T).mm(y.T)
            nakagami = NonCentralNakagami(mu_x, var_x)
            finsler[i1, i2] = nakagami.expectation()

    cs = plt.contour(finsler.detach().numpy(), (1,))
    plt.close()
    pp = cs.collections[0].get_paths()[0].vertices / (size) - [0.5, 0.5] + point.detach().numpy()
    codes = Path.CURVE3 * np.ones(len(pp), dtype=Path.code_type)
    codes[0] = codes[-1] = Path.MOVETO
    return Path(pp, codes)


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


def isPSD(A, tol=1e-8):
    E = np.linalg.eigvalsh(A)
    return np.all(E > -tol)


# visualisation


def grid_latent(X, model, n_grid=25):
    # X = X.detach().numpy()
    Xmin, Xmax = np.min(X, axis=0), np.max(X, axis=0)
    ux, uy = np.meshgrid(np.linspace(Xmin[0], Xmax[0], n_grid), np.linspace(Xmin[1], Xmax[1], n_grid))
    xx = torch.tensor([ux.flatten(), uy.flatten()]).t()
    return xx


def scatterplot_matrix(xx, model, **kwargs):
    """Plots a scatterplot matrix of subplots for font data."""
    import matplotlib

    data = model.embed(xx.squeeze())[0].detach().numpy()
    num_img = data.shape[0]
    data = data.reshape((num_img, 2, -1))
    numdata = int(np.sqrt(data.shape[0]))

    fig, axes = plt.subplots(nrows=numdata, ncols=numdata, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    var = np.log10(model.embed(xx)[1].detach().numpy())

    def norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    cmap = matplotlib.cm.get_cmap("RdBu")

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # inside = int(data.shape[2]/2)
    for j in range(numdata):
        for i in range(numdata):
            x = i + (j % numdata) * numdata
            # axes[numdata-j-1,i].plot(data[x,0,:-1], data[x,1,:-1], color=cmap(norm(var)[x]), s=0.5, **kwargs)
            axes[numdata - j - 1, i].plot(data[x, 0, :], data[x, 1, :], c=cmap(norm(var)[x]), **kwargs)
    return fig


def indicatrices(model, points, mode="riemann"):
    """Draw the indicatrix of functions using a range of vectors.
    ---
    inputs: - cov: qxq covariance matrix, with cov = var[J]
            - mean: qxq matrix, with mean = E[J].T@E[J]
            - points: points for which we want to plot the indicatrix
            - scale: reduce the size of the indicatrix for visualisation purposes
    """
    from geoml.stats import NonCentralNakagami

    # vectors = torch.tensor(vectors, dtype=torch.float32)
    # mus, vars = torch.squeeze(mus.to(torch.float32)), torch.squeeze(vars.to(torch.float32))
    mus, vars = model.jacobian_posterior(points)
    mus, vars = mus, vars
    n_of_points, D = mus.size(0), mus.size(2)
    paths, remove_index = [], []
    vector_values = torch.empty((n_of_points))

    for nn in range(n_of_points):
        metric = mus[nn].mm(mus[nn].T) + D * vars[nn]
        alpha, vec_size = automated_scaling(metric)
        coeff = 1.1
        vectors = torch.linspace(-coeff * alpha, coeff * alpha, vec_size)
        vector_values[nn] = (2 * coeff * alpha) ** 2
        indicatrix = torch.empty((vec_size, vec_size))

        for i1, y1 in enumerate(vectors):
            for i2, y2 in enumerate(vectors):
                y = torch.unsqueeze(torch.tensor([y1, y2]), 0)  # random vectors
                var_x = y.mm(vars[nn]).mm(y.T)
                mu_x = y.mm(mus[nn]).mm(mus[nn].T).mm(y.T)

                if mode == "riemann":
                    indicatrix[i1, i2] = mu_x + D * var_x
                elif mode == "finsler":
                    nakagami = NonCentralNakagami(mu_x, var_x)
                    indicatrix[i1, i2] = nakagami.expectation()

        figc, axc = plt.subplots(1, 1)
        cs = axc.contour(indicatrix.detach().numpy(), (1.0,))
        plt.close(figc)
        sgm = cs.collections[0].get_segments()[0]
        eps = 1e-2
        if (
            (abs(sgm[0, 0] - sgm[-1, 0]) < eps)
            and (abs(sgm[0, 1] - sgm[-1, 1]) < eps)
            and len(cs.collections[0].get_segments()) == 1
        ):
            pp = cs.collections[0].get_paths()[0].vertices / (vec_size) - [0.5, 0.5] + points[nn].detach().numpy()
            codes = Path.CURVE3 * np.ones(len(pp), dtype=Path.code_type)
            codes[0] = codes[-1] = Path.MOVETO
            path = Path(pp, codes)
            paths.append(path)
        else:
            print("issue with points {}, contour broken".format(nn))
            remove_index.append(nn)
            paths.append(np.nan)
        # fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
        # axs.contour(indicatrix[nn,:,:].detach().numpy(), (1,), colors='orange', linewidths=2, alpha=0.7)
        # axs.imshow(indicatrix[nn,:,:].detach().numpy())
        # axs.set_title('{} for point: {}, ax-ay:{:.2f}-{:.2f}'.format(mode, nn, alphax, alphay))
        # plt.show()

    # for nn in range(n_of_points):
    #     figc, axc = plt.subplots(1, 1)
    #     cs = axc.contour(indicatrix[nn].detach().numpy(), (scale,))
    #     plt.close(figc)
    #     # plt.contour(indicatrix[nn].detach().numpy(), (scale,))
    #     # print(nn, len(cs.collections[0].get_segments()))
    #     sgm = cs.collections[0].get_segments()[0]
    #     eps = 1e-2
    #     if (abs(sgm[0,0] - sgm[-1,0]) < eps) and (abs(sgm[0,1] - sgm[-1,1]) < eps) and  \\
    #        len(cs.collections[0].get_segments())==1:
    #         pp = cs.collections[0].get_paths()[0].vertices/(vector_size[nn]) - [0.5,0.5] + points[nn].detach().numpy()
    #         codes = Path.CURVE3 * np.ones(len(pp), dtype=Path.code_type)
    #         codes[0] = codes[-1] = Path.MOVETO
    #         path = Path(pp, codes)
    #         paths.append(path)
    #     else:
    #         print('issue with points {}, contour broken'.format(nn))
    #         remove_index.append(nn)
    #         paths.append(np.nan)
    # vector_values = np.delete(vector_values, remove_index, axis=0)
    # points = np.delete(points.detach().numpy(), remove_index, axis=0)
    return paths, points, vector_values


# def stack_contour(ax):
#     ax.contour(finsler, (1,), colors='tab:orange', linewidths=1, linestyles='dashed')


def PolyArea(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def volume(model, points, mode):
    scale = 1
    paths, points, vec_values = indicatrices(model, points, mode=mode)
    volume_hausdorff = np.empty(len(paths))
    for i, path in enumerate(paths):
        if (path != path) or (PolyArea(path.vertices) < 0.01):
            # print('--------',torch.tensor(points[i]))
            # point = torch.unsqueeze(torch.tensor(points[i]), 0)
            # metric = model.metric(point).detach().numpy()
            # volume_hausdorff[i] = np.sqrt(np.linalg.det(metric))
            volume_hausdorff[i] = volume_hausdorff[i - 1]
            # volume_hausdorff[i] = np.nan
        else:
            volume_indicatrix = PolyArea(path.vertices) * (vec_values[i]) / scale
            volume_hausdorff[i] = np.pi / volume_indicatrix
    return volume_hausdorff, paths, points


# def volume_compare(model, xx):
#     vol_idx, _, points = volume_finsler(model, xx)
#     vol_det = np.sqrt(np.linalg.det(model.metric(points).detach().numpy()))
#     return vol_det - vol_idx


def volume_heatmap(ax, model, X, mode, n_grid=25, log=True):
    # map that represents the log(variance)
    if torch.is_tensor(X):
        X = X.detach().numpy()
    Xmin, Xmax = np.min(X, axis=0), np.max(X, axis=0)
    ux, uy = np.meshgrid(np.linspace(Xmin[0], Xmax[0], n_grid), np.linspace(Xmin[1], Xmax[1], n_grid))
    xx = torch.tensor([ux.flatten(), uy.flatten()]).t()
    if mode == "variance":
        # map = np.diag(model.embed_old2(xx)[1].detach().numpy()) # variance GPLVM
        map = model.embed(xx)[1].detach().numpy()
    elif mode == "vol_riemann":
        map = np.sqrt(np.linalg.det(model.metric(xx).detach().numpy()))  # volume Riemann
    elif mode == "vol_riemann2":
        map, paths, xx = volume(model, xx, mode="riemann")  # volume Riemann
    elif mode == "vol_finsler":
        map, paths, xx = volume(model, xx, mode="finsler")
        # for path in paths:
        #     patch = PathPatch(path, linewidth=1, edgecolor='tab:orange', fill=False)
        #     ax.add_patch(patch)
    elif mode == "diff":
        map_riemann, __, __ = volume(model, xx, mode="riemann")
        map_finsler, paths, __ = volume(model, xx, mode="finsler")
        map = np.abs((map_riemann - map_finsler)) / map_riemann
        print("number of negative elemnts", np.sum(map < 0))
        print("min map:", np.min(map))
        map[map <= 1e-8] = 1e-8

        # idx_high = np.where(map<-0.5)[0]
        # for i in idx_high:
        #     patch = PathPatch(paths[i], linewidth=1, edgecolor='tab:orange', fill=False)
        #     ax.add_patch(patch)
        # ax.scatter(xx[idx_high,0], xx[idx_high,1], color='red', s=2)
        # indicatrixs = indicatrices(model, xx[idx_high], mode='finsler')[-1]
        # for indicatrix in indicatrixs:
        #     figc, axc = plt.subplots(1, 1)
        #     axc.contour(indicatrix.detach().numpy(), (1,), colors='tab:orange', linewidths=1, linestyles='dashed')
        #     axc.imshow(indicatrix.detach().numpy())
        #     plt.show()

    if torch.is_tensor(xx):
        xx = xx.detach().numpy()
    if log:
        map = np.log10(map)

    grid = griddata(xx, map, (ux, uy), method="nearest")
    im = ax.imshow(
        grid,
        extent=(Xmin[0], Xmax[0], Xmin[1], Xmax[1]),
        origin="lower",
        cmap="RdBu",
        aspect="auto",
        interpolation="bicubic",
    )
    # by definition the variance matrix is isotropic,
    # so we have the same element on the diagonal
    # logmap = np.log(map)
    # out = np.asarray(map).reshape((n_grid, n_grid))
    # im = ax.imshow(out, interpolation='bicubic',
    #                extent=(Xmin[0], Xmax[0], Xmin[1], Xmax[1]),
    #                aspect='auto', origin='lower', cmap='RdBu')

    return ax, im, xx


def contour_riemann(finsler, riemann, vectors, out_dir):
    scale = len(vectors) / (np.max(vectors) - np.min(vectors))
    center = (int(len(vectors) / 2), int(len(vectors) / 2))
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    axs.contour(finsler, (1,), colors="tab:orange", linewidths=1, linestyles="dashed")
    axs.add_patch(ellipse_draw(riemann, center, scale, linewidth=1, edgecolor="tab:blue", linestyle="dashed"))
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc="tab:orange", ec="white", alpha=0.7, linewidth=4)
    proxy2 = plt.Rectangle((0, 0), 1, 1, fc="tab:blue", ec="white", alpha=0.7, linewidth=4)
    axs.legend(handles=[proxy1, proxy2], labels=["finsler", "riemann"])
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal")
    fig.savefig(os.path.join(out_dir, "{}.png".format("riemann")), dpi=fig.dpi)


def ellipse_draw(matrix, center, scale, **kwargs):
    """Get matplotlib patch of an ellipse draw from a psd 2x2 matrix."""
    if torch.is_tensor(matrix):
        matrix = torch.squeeze(matrix).detach().numpy()
    if torch.is_tensor(center):
        center = torch.squeeze(center).detach().numpy()
    width, height, angle = ellipse_params(matrix)
    return Ellipse(center, width=scale * width, height=scale * height, angle=angle, fill=False, **kwargs)


def ellipse_params(matrix):
    """Get the params of an ellipse for the matplotlib func: ellpse_draw.
    ---
    input: 2x2 psd matrix
    output: semi-axis (width and height), angles (in degrees)"""
    eigvalues, eigvectors = np.linalg.eig(matrix)
    idx = np.argsort(eigvalues)
    eigvalues, eigvectors = eigvalues[idx], eigvectors[:, idx]
    height, width = 1 / np.sqrt(eigvalues[1]), 1 / np.sqrt(eigvalues[0])
    (cos_theta, sin_theta) = eigvectors[:, 1]
    theta = np.arctan(sin_theta / cos_theta)
    return 2 * width, 2 * height, -np.rad2deg(theta)


def indicatrices_on_geodesic(ax, model, points, mode, color):
    """Plot both Riemann and Finsler indicatrices for different points"""
    paths = indicatrices(model, points, mode=mode)[0]
    for path in paths:
        patch = PathPatch(path, linewidth=1, edgecolor=color, fill=False)
        ax.add_patch(patch)
    return ax
    # fig.savefig(os.path.join(out_dir,'{}.png'.format('riemann')), dpi=fig.dpi)


def automated_scaling(metric):
    """scale the vector to compute the indicatrix"""
    if torch.is_tensor(metric):
        metric = torch.squeeze(metric).detach().numpy()
    eigvalues, _ = np.linalg.eig(metric)
    ratio = np.max(eigvalues) / np.min(eigvalues)
    # this is awful and arbitrary
    if ratio < 10:
        vec_size = 8
    elif 10 < ratio < 50:
        vec_size = 16
    elif 50 < ratio < 200:
        vec_size = 32
    elif 200 < ratio:
        vec_size = 64
    else:
        vec_size = 16
    return 1 / np.sqrt(np.min(eigvalues)), vec_size


def automated_scaling2(metric):
    """scale the vector ... Not really improving anything"""
    if torch.is_tensor(metric):
        metric = torch.squeeze(metric).detach().numpy()
    eigvalues, eigvectors = np.linalg.eig(metric)
    idx = np.argsort(eigvalues)
    eigvalues, eigvectors = eigvalues[idx], eigvectors[:, idx]
    long_size, short_size = 1 / np.sqrt(eigvalues[0]), 1 / np.sqrt(eigvalues[1])
    (cos_theta, sin_theta) = eigvectors[:, 1]
    theta = np.rad2deg(np.arctan(sin_theta / cos_theta))
    if (theta % 180) < 45 or (theta % 180) > 135:
        return long_size, 2 * short_size
    elif 45 < (theta % 180) < 135 or (theta % 180) > 135:
        return 2 * short_size, long_size


# def indicatrices_on_geodesic_old(ax, model, points, scale=1.0):
# """ Plot both Riemann and Finsler indicatrices for different points """
# n_of_points = points.size(0)
# mus, covs = model.jacobian(points)
# metrics_riemann = model.data.shape[1]*covs + torch.bmm(mus,mus.transpose(1,2))
# vectors = np.linspace(-1, 1, 64)
# len_vec = (np.max(vectors)-np.min(vectors))
# for i in range(n_of_points):
#     ax.add_patch(ellipse_draw(metrics_riemann[i], points[i], scale=scale/len_vec,
#                               linewidth=1, edgecolor='tab:blue'))
#     # need scale because not the same scale between finsler and riemann
# return ax


# BECAUSE THERE ARE ISSUES WITH THE GPLVM CODE :(
def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q * xdiag * Q.T


def _getPs(A, W=None):
    W05 = np.matrix(W**0.5)
    return W05.I * _getAplus(W05 * A * W05) * W05.I


def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)


def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n)
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk
