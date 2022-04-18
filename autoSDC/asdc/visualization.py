import os
import numpy as np

import ternary
from ternary.helpers import simplex_iterator

import mpltern
from mpltern.ternary.datasets import get_triangular_grid

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import warnings

warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import asdc.analyze


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def plot_iv(I, V, figpath="iv.png"):
    plt.plot(np.log10(np.abs(I)), V)
    plt.xlabel("log current")
    plt.ylabel("voltage")
    plt.savefig(figpath, bbox_inches="tight")
    plt.clf()
    plt.close()
    return


def plot_lpr(I, V, figpath="lpr.png"):
    plt.plot(I, V)
    plt.axvline(0, color="k")
    plt.xlabel("current")
    plt.ylabel("potential")
    plt.savefig(figpath, bbox_inches="tight")
    plt.clf()
    plt.close()
    return


def plot_vi(I, V, figpath="iv.png"):
    plt.plot(V, np.log10(np.abs(I)))
    plt.ylabel("log current")
    plt.xlabel("voltage")
    plt.savefig(figpath, bbox_inches="tight")
    plt.clf()
    plt.close()
    return


def plot_v(t, V, figpath="v.png"):
    n = min(len(t), len(I))
    plt.plot(t[:n], V[:n])
    plt.xlabel("time (s)")
    plt.ylabel("voltage")
    plt.savefig(figpath, bbox_inches="tight")
    plt.clf()
    plt.close()
    return


def plot_i(t, I, figpath="i.png"):
    n = min(len(t), len(I))
    plt.plot(t[:n], I[:n])
    plt.xlabel("time (s)")
    plt.ylabel("current (A)")
    plt.savefig(figpath, bbox_inches="tight")
    plt.clf()
    plt.close()
    return


def plot_cv(V, I, segment=None, segments=[2, 3], figpath="cv.png"):

    for s in segments:
        plt.plot(V[segment == s], I[segment == s], label=s)

    plt.xlabel("potential (V)")
    plt.ylabel("current (A)")
    plt.legend()
    plt.savefig(figpath, bbox_inches="tight")
    plt.clf()
    plt.close()
    return


def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))


def combi_plot(show_flat=False):
    """scatter plot visualizations on a 3-inch combi wafer.
    coordinate system is specified in mm
    """
    R = 76.2 / 2
    c = make_circle(R)  # 3 inch wafer --> 76.2 mm diameter
    if show_flat:
        sel = c[:, 1] > -35
        c = c[sel]
    plt.plot(c[:, 0], c[:, 1], color="k")


def scatter_wafer(X, y, label=None, figpath="wafer_plot.png"):
    fig, axes = plt.subplots(figsize=(4.5, 4))
    combi_plot(show_flat=True)

    p = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Blues", edgecolors="k")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")

    cbar = colorbar(p)
    cbar.set_label(label)
    plt.savefig(figpath, bbox_inches="tight")
    plt.clf()
    plt.close()
    return


def plot_open_circuit(current, potential, segment, figpath="open_circuit.png"):
    plt.figure(figsize=(4, 5))

    model = asdc.analyze.extract_open_circuit_potential(
        current, potential, segment, return_model=True
    )
    plt.plot(-model.data, model.userkws["x"], color="b")
    plt.plot(-model.best_fit, model.userkws["x"], c="r", linestyle="--", alpha=0.5)
    plt.axhline(model.best_values["peak_loc"], c="k", linestyle="--", alpha=0.5)

    plt.xlabel("log current (log (A)")
    plt.ylabel("potential (V)")
    plt.savefig(figpath, bbox_inches="tight")
    plt.clf()
    plt.close()
    return


def plot_ocp_model(x, y, ocp, gridpoints, model, query_position, figure_path=None):

    N, _ = gridpoints.shape
    w = int(np.sqrt(N))
    mu_y, var_y = model.predict_y(gridpoints)

    plt.figure(figsize=(5, 4))
    combi_plot()

    plt.scatter(x, y, c=ocp, edgecolors="k", cmap="Blues")
    plt.axis("equal")

    cmap = plt.cm.Blues
    colors = Normalize(vmin=mu_y.min(), vmax=mu_y.max(), clip=True)(mu_y.flatten())
    # colors = mu_y.flatten()
    c = cmap(colors)
    a = Normalize(var_y.min(), var_y.max(), clip=True)(var_y.flatten())
    # c[...,-1] = 1-a

    c[np.sqrt(np.square(gridpoints).sum(axis=1)) > 76.2 / 2, -1] = 0
    c = c.reshape((w, w, 4))

    extent = (
        np.min(gridpoints),
        np.max(gridpoints),
        np.min(gridpoints),
        np.max(gridpoints),
    )
    im = plt.imshow(c, extent=extent, origin="lower", cmap=cmap)
    cbar = plt.colorbar(im, extend="both")
    plt.clim(mu_y.min(), mu_y.max())

    plt.scatter(query_position[0], query_position[1], c="none", edgecolors="r")
    plt.axis("equal")

    if figure_path is not None:
        plt.savefig(figure_path, bbox_inches="tight")
        plt.clf()


def mpltern_scatter(
    df,
    y="FWHM",
    components=["Al", "Ti", "Ni"],
    y_name=None,
    fig=None,
    panel=None,
    tripcolor=False,
    cmap=None,
    shading="flat",
    vlimits=(None, None),
    use_colorbar=True,
):
    """ mpltern scatter plot helper """
    # reorder composition by components

    X = df.loc[:, components]

    if y_name is None:
        y_name = y

    if type(y) is str:
        y = df[y]

    if fig is None:
        fig = plt.gcf()
    if panel is None:
        ax = plt.subplot(projection="ternary")
    elif type(panel) == matplotlib.gridspec.SubplotSpec:
        ax = fig.add_subplot(panel, projection="ternary")
    else:
        ax = panel

    ax.tick_params(labelrotation="horizontal")

    # draw gridlines
    t, l, r = get_triangular_grid()
    ax.triplot(t, l, r, color="k", zorder=-100, alpha=0.5, linewidth=0.5)

    if not tripcolor:
        s = ax.scatter(
            X.iloc[:, 0],
            X.iloc[:, 1],
            X.iloc[:, 2],
            c=y,
            cmap=cmap,
            edgecolors="k",
            vmin=vlimits[0],
            vmax=vlimits[1],
        )
    else:
        s = ax.tripcolor(
            X.iloc[:, 0],
            X.iloc[:, 1],
            X.iloc[:, 2],
            y,
            shading="flat",
            cmap=cmap,
            vmin=vlimits[0],
            vmax=vlimits[1],
        )

    axis_order = ["tlabel", "llabel", "rlabel"]
    axis_labels = {a: component for a, component in zip(axis_order, components)}

    ax.set(**axis_labels)

    # draw the colorbar on an inset
    if use_colorbar:
        cax = ax.inset_axes([1.075, 0.1, 0.025, 0.9], transform=ax.transAxes)
        colorbar = fig.colorbar(s, cax=cax, label=y_name)

    # normalize ticks and remove zero tick label
    for axis in (ax.taxis, ax.laxis, ax.raxis):
        axis.set_label_rotation_mode("horizontal")
        axis.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])

    return ax


def ternary_scatter(
    composition,
    value,
    components=["Ni", "Al", "Ti"],
    cmap="Blues",
    label=None,
    cticks=None,
    s=50,
):
    """ python-ternary version """

    scale = 1
    grid = plt.GridSpec(10, 1, wspace=1, hspace=3)
    ax = plt.subplot(grid[:9, :])

    filtered = value[np.isfinite(value)]
    vmin, vmax = filtered.min(), filtered.max()

    figure, tax = ternary.figure(scale=scale, ax=ax)
    figure.set_size_inches(6, 6)
    s = tax.scatter(
        composition,
        marker="o",
        c=value,
        cmap=cmap,
        edgecolors="k",
        vmin=vmin,
        vmax=vmax,
        s=s,
    )
    # s = tax.scatter(composition, marker='o', c=value, cmap=cmap, edgecolors='k', s=50)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=0.1, color="k")
    tax.ticks(axis="lbr", linewidth=1, multiple=0.2, tick_formats="%0.01f", offset=0.02)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")

    tax.right_corner_label(components[0], fontsize=18, offset=-0.1)
    tax.top_corner_label(components[1], fontsize=18)
    tax.left_corner_label(components[2], fontsize=18, offset=0.2)
    ax.axis("equal")

    ax = plt.subplot(grid[9:, :])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # norm = matplotlib.colors.Normalize(vmin=-0.9, vmax=value.max())
    cb1 = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="horizontal", label=label
    )

    if cticks is not None:
        cb1.set_ticks(cticks)
        # cb1.set_tick_labels([-.85, -.7, -0.5])

    plt.subplots_adjust()
    plt.tight_layout()
    return tax


def ternary_scatter_sub(
    composition,
    value,
    components=["Ni", "Al", "Ti"],
    cmap="Blues",
    label=None,
    cticks=None,
    s=50,
    ax=None,
):
    scale = 1
    if ax is None:
        fig, ax = plt.subplots()
    # grid = plt.GridSpec(10, 1, wspace=1, hspace=3)
    # ax = plt.subplot(grid[:9, :])

    filtered = value[np.isfinite(value)]
    vmin, vmax = filtered.min(), filtered.max()

    figure, tax = ternary.figure(scale=scale, ax=ax)
    # figure.set_size_inches(6, 6)
    s = tax.scatter(
        composition,
        marker="o",
        c=value,
        cmap=cmap,
        edgecolors="k",
        vmin=vmin,
        vmax=vmax,
        s=s,
    )
    # s = tax.scatter(composition, marker='o', c=value, cmap=cmap, edgecolors='k', s=50)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=0.1, color="k")
    tax.ticks(axis="lbr", linewidth=1, multiple=0.2, tick_formats="%0.01f", offset=0.02)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")

    # tax.right_corner_label(components[0], fontsize=18, offset=0.1)
    # tax.top_corner_label(components[1], fontsize=18, offset=0.1)
    # tax.left_corner_label(components[2], fontsize=18, offset=0.1)
    tax.bottom_axis_label(components[0], fontsize=18, offset=0.1)
    tax.right_axis_label(components[1], fontsize=18, offset=0.1)
    tax.left_axis_label(components[2], fontsize=18, offset=0.1)

    ax.axis("equal")

    # ax = plt.subplot(grid[9:,:])
    # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # # norm = matplotlib.colors.Normalize(vmin=-0.9, vmax=value.max())
    # cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal', label=label)

    # if cticks is not None:
    #     cb1.set_ticks(cticks)
    #     # cb1.set_tick_labels([-.85, -.7, -0.5])

    plt.subplots_adjust()
    plt.tight_layout()
    return tax


def ternary_heatmap(
    model,
    components=["Ni", "Al", "Ti"],
    cmap="Blues",
    label=None,
    cticks=None,
    scale=10,
    plot_var=False,
    nticks=5,
    sample_posterior=False,
):

    keys = [k for k in simplex_iterator(scale)]
    # o_mu, o_var = model.predict_f(np.array(keys).astype(float) / scale)
    if sample_posterior:
        o_mu = model.predict_f_samples(
            np.array(keys).astype(float)[:, :2] / scale, 1
        ).squeeze()
    else:
        o_mu, o_var = model.predict_f(np.array(keys).astype(float)[:, :2] / scale)
    if plot_var:
        v = o_var
    else:
        v = o_mu

    tdata = {key: v for key, v in zip(keys, v.flat)}

    grid = plt.GridSpec(10, 1, wspace=1, hspace=3)
    ax = plt.subplot(grid[:9, :])

    vmin, vmax = v.min(), v.max()

    figure, tax = ternary.figure(scale=scale, ax=ax)
    tax.set_axis_limits({"b": (0, 1.0), "l": (0, 1.0), "r": (0, 1.0)})
    tax.get_ticks_from_axis_limits(multiple=scale / nticks)

    figure.set_size_inches(6, 6)
    s = tax.heatmap(tdata, scale, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False)

    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=scale / nticks, color="k", alpha=0.5)
    tax.clear_matplotlib_ticks()
    tax.set_custom_ticks(tick_formats="%0.1f", offset=0.02)
    tax.get_axes().axis("off")

    tax.right_corner_label(components[0], fontsize=18, offset=-0.1)
    tax.top_corner_label(components[1], fontsize=18)
    tax.left_corner_label(components[2], fontsize=18, offset=0.2)
    ax.axis("equal")

    ax = plt.subplot(grid[9:, :])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # norm = matplotlib.colors.Normalize(vmin=-0.9, vmax=value.max())
    cb1 = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="horizontal", label=label
    )

    if cticks is not None:
        cb1.set_ticks(cticks)
        # cb1.set_tick_labels([-.85, -.7, -0.5])

    plt.subplots_adjust()
    plt.tight_layout()
    return tax
