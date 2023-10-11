import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata

from finsler.utils.data import make_sphere_points, make_sphere_surface


def template_colormap(ax, n_grid, cmap):
    ux, uy = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
    xx = np.stack((ux.flatten(), uy.flatten())).T
    map = np.sin(xx[:, 0]) * np.cos(xx[:, 1])
    grid = griddata(xx, map, (ux, uy), method="nearest")
    im = ax.imshow(grid, extent=(0, 1, 0, 1), origin="lower", cmap=cmap, aspect="auto", interpolation="bicubic")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax, im


def starfish():
    Y = make_sphere_points(500)
    ax.set_box_aspect((np.ptp(Y[:, 0]), np.ptp(Y[:, 1]), np.ptp(Y[:, 2])))
    XS, YS, ZS = make_sphere_surface()  # for illustration
    # ax.plot_surface(XS, YS, ZS, shade=False, rstride=2, cstride=2, linewidth=0, color='gray', alpha=0.9)
    ax.plot_surface(XS, YS, ZS, color="gray")
    # ax.plot_wireframe(XS, YS, ZS, rstride=2, cstride=2, linewidth=0.4, color='white', alpha=0.5)
    ax.scatter3D(
        Y[:, 0], Y[:, 1], Y[:, 2], label="observed data", marker="o", edgecolors="black", s=1
    )  # observed data points
    ax.grid(False)
    ax.axis("off")
    return ax


def heatmap_sphere(ax):
    from finsler.utils.data import make_sphere_surface

    # coordinates
    X, Y, Z = make_sphere_surface()

    # colormap
    C = np.linspace(-5, 5, Z.size).reshape(Z.shape)
    scamap = plt.cm.ScalarMappable(cmap="inferno")
    fcolors = scamap.to_rgba(C)

    return ax, fcolors, (X, Y, Z), scamap


if __name__ == "__main__":

    # cmap_sns = sns.color_palette("flare", as_cmap=True)
    # fig = plt.figure(1)
    # ax = plt.axes()
    # ax, im = template_colormap(ax, n_grid=10, cmap=cmap_sns)
    # fig.colorbar(im)
    # plt.title('Template title')
    # plt.savefig('debug/plots/colormap.svg', format='svg')
    # plt.show()

    fig = plt.figure(2)
    ax = plt.axes(projection="3d")
    ax = starfish()
    plt.legend()
    plt.title("Sphere")
    # plt.savefig('debug/plots/sphere.svg', format='svg')
    plt.show()

    # fig = plt.figure(2)
    # ax = plt.axes(projection='3d')
    # ax, fcolors, (X,Y,Z), scamap = heatmap_sphere(ax)
    # ax.plot_surface(X,Y,Z, facecolors=fcolors, cmap='inferno')
    # fig.colorbar(scamap)
    # plt.title('Sphere')
    # plt.savefig('debug/plots/sphere_heatmap.svg', format='svg')
    # plt.show()
