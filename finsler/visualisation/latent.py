import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.interpolate import griddata

from finsler.distributions import NonCentralNakagami


def automated_scaling(metric):
    """scale the vector to compute the indicatrix"""
    if torch.is_tensor(metric):
        metric = torch.squeeze(metric).detach().numpy()
    eigvalues, _ = np.linalg.eig(metric)
    vec_size = 64
    return 1 / np.sqrt(np.min(eigvalues)), vec_size


def PolyArea(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def indicatrices(model, points, mode="riemann"):
    """Draw the indicatrix of functions using a range of vectors.
    ---
    inputs: - cov: qxq covariance matrix, with cov = var[J]
            - mean: qxq matrix, with mean = E[J].T@E[J]
            - points: points for which we want to plot the indicatrix
            - scale: reduce the size of the indicatrix for visualisation purposes
    """

    # vectors = torch.tensor(vectors, dtype=torch.float32)
    # mus, vars = torch.squeeze(mus.to(torch.float32)), torch.squeeze(vars.to(torch.float32))
    mus, vars = model.jacobian_posterior(points)
    mus, vars = mus, vars
    n_of_points, D = mus.size(0), mus.size(2)
    paths, remove_index = [], []
    vector_values = torch.empty((n_of_points))

    for nn in range(n_of_points):
        metric = mus[nn].mm(mus[nn].T) + D * vars[nn]
        alpha, vec_size = automated_scaling(metric)  # needed to get good resolution for contour
        coeff = 2.0 * 1.1
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
        plt.close(figc)  # problem saving polygons if figc not close
        polygon = cs.allsegs[0]  # getting our indicatrix (polygon)
        if len(polygon) != 1:
            # plt.imshow(indicatrix.detach().numpy())
            # plt.contour(indicatrix.detach().numpy(), (1.0,))
            # plt.title('mode: {}, point #{}'.format(mode, nn))
            # plt.show()
            print("issue with points {}/{}, contour broken".format(nn, n_of_points))
            remove_index.append(nn)
            paths.append(np.nan)
        else:
            # center, normalise and place indicatrix
            pp = polygon[0] / vec_size - [0.5, 0.5] + points[nn].detach().numpy()
            codes = Path.CURVE3 * np.ones(len(pp), dtype=Path.code_type)
            codes[0] = codes[-1] = Path.MOVETO  # we close the polygon
            path = Path(pp, codes)
            paths.append(path)
    return paths, points, vector_values


def volume(model, points, mode):
    scale = 1
    paths, points, vec_values = indicatrices(model, points, mode=mode)
    volume_hausdorff = np.empty(len(paths))
    for i, path in enumerate(paths):
        if (path != path) or (PolyArea(path.vertices) < 0.01):
            volume_hausdorff[i] = volume_hausdorff[i - 1]
        else:
            volume_indicatrix = PolyArea(path.vertices) * (vec_values[i]) / scale
            volume_hausdorff[i] = np.pi / volume_indicatrix
    return volume_hausdorff, paths, points


def volume_heatmap(ax, model, X, mode, n_grid=20, log=True, vmin=None, vmax=None, with_indicatrix=False):
    print(end="/n")
    if with_indicatrix:
        if mode in ["variance", "vol_riemann"]:
            print("You can only plot the indocatrices with mode: vol_riemann2 or vol_finsler")
    cmap_sns = sns.color_palette("flare", as_cmap=True)
    # map that represents the log(variance)
    print("Computing volume_heatmap...")
    if torch.is_tensor(X):
        X = X.detach().numpy()
    Xmin, Xmax = np.min(X, axis=0) - 1, np.max(X, axis=0) + 1
    ux, uy = np.meshgrid(np.linspace(Xmin[0], Xmax[0], n_grid), np.linspace(Xmin[1], Xmax[1], n_grid))
    xx = torch.stack((torch.tensor(ux.flatten()), torch.tensor(uy.flatten()))).t()
    # xx = torch.tensor([ux.flatten(), uy.flatten()]).t()
    if mode == "variance":
        # map = np.diag(model.embed_old2(xx)[1].detach().numpy()) # variance GPLVM
        map = model.embed(xx, full_cov=False)[1].detach().numpy()
    elif mode == "vol_riemann":
        map = np.sqrt(np.linalg.det(model.metric(xx).detach().numpy()))  # volume Riemann
    elif mode == "vol_riemann2":
        map, paths, xx = volume(model, xx, mode="riemann")  # volume Riemann
    elif mode == "vol_finsler":
        map, paths, xx = volume(model, xx, mode="finsler")
    elif mode == "diff":
        map_riemann, __, __ = volume(model, xx, mode="riemann")
        map_finsler, paths, __ = volume(model, xx, mode="finsler")
        map = np.abs((map_riemann - map_finsler)) / map_riemann
        print("number of negative elemnts", np.sum(map < 0))
        print("min map:{}, max map:{}".format(np.min(map), np.max(map)))
        map[map <= 1e-8] = 1e-8

    if torch.is_tensor(xx):
        xx = xx.detach().numpy()
    if log:
        map = np.log10(map)

    grid = griddata(xx, map, (ux, uy), method="nearest")

    if with_indicatrix:
        for path in paths:
            patch = PathPatch(path, linewidth=1, edgecolor="tab:orange", fill=False)
            ax.add_patch(patch)

    im = ax.imshow(
        grid,
        # extent=(Xmin[0], Xmax[0], Xmin[1], Xmax[1]),
        extent=(-4, 4, -4, 4),
        origin="lower",
        cmap=cmap_sns,
        aspect="auto",
        interpolation="bicubic",
        vmin=vmin,
        vmax=vmax,
    )
    return ax, im, map, xx


def plot_indicatrices(ax, model, X, n_grid=10):
    Xmin, Xmax = np.min(X, axis=0) - 1, np.max(X, axis=0) + 1
    ux, uy = np.meshgrid(np.linspace(Xmin[0], Xmax[0], n_grid), np.linspace(Xmin[1], Xmax[1], n_grid))
    xx = torch.stack((torch.tensor(ux.flatten()), torch.tensor(uy.flatten()))).t()
    paths_riemann, _, _ = indicatrices(model, xx, mode="riemann")
    paths_finsler, _, _ = indicatrices(model, xx, mode="finsler")

    paths_riemann = [path for path in paths_riemann if str(path) != "nan"]
    paths_finsler = [path for path in paths_finsler if str(path) != "nan"]
    print("only {} indicatrices were plotted for Finsler".format(len(paths_finsler)))

    print("plot indicatrices for Riemann")
    for path in paths_riemann:
        pr = PathPatch(path, edgecolor="tab:orange", fill=False, lw=1, alpha=0.5)
        ax.add_patch(pr)
    print("plot indicatrices for Finsler")
    for path in paths_finsler:
        pf = PathPatch(path, edgecolor="rebeccapurple", fill=False, lw=1, alpha=0.5)
        ax.add_patch(pf)
    return ax


def plot_indicatrices_along_geodesic(ax, model, curve_riemann, curve_finsler, n_grid=5):

    xxr = curve_riemann[:: int(curve_riemann.shape[0] / n_grid) - 1]
    xxf = curve_finsler[:: int(curve_finsler.shape[0] / n_grid) - 1]

    paths_riemann, _, _ = indicatrices(model, xxr, mode="riemann")
    paths_finsler, _, _ = indicatrices(model, xxf, mode="finsler")

    paths_riemann = [path for path in paths_riemann if str(path) != "nan"]
    paths_finsler = [path for path in paths_finsler if str(path) != "nan"]
    print("only {} indicatrices were plotted for Finsler".format(len(paths_finsler)))

    print("plot indicatrices for Riemann")
    for path in paths_riemann:
        pr = PathPatch(path, edgecolor="tab:orange", fill=False, lw=1, alpha=0.5)
        ax.add_patch(pr)
    print("plot indicatrices for Finsler")
    for path in paths_finsler:
        pf = PathPatch(path, edgecolor="rebeccapurple", fill=False, lw=1, alpha=0.5)
        ax.add_patch(pf)
    return ax
