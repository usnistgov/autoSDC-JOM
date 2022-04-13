import os
import glob
import json
import numpy as np
import pandas as pd

import gpflow

from asdc import analyze
from asdc import visualization


def load_dataset(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.json"))
    df = None
    for idx, datafile in enumerate(files):

        with open(datafile, "r") as ff:
            d = json.load(ff)

        _df = pd.DataFrame.from_dict(d, orient="index").T

        if df is None:
            df = _df
        else:
            df = pd.concat((df, _df))

    df = df.reset_index(drop=True)
    return df


def gp_select(data_dir, plot_model=True, idx=0):
    df = load_dataset(data_dir)
    ocp = [
        analyze.extract_open_circuit_potential(row.current, row.potential, row.segment)
        for idx, row in df.iterrows()
    ]
    x = df.position_combi.apply(lambda x: x[0])
    y = df.position_combi.apply(lambda x: x[1])

    # fit a GP regression model
    X = np.vstack((x.values, y.values)).T
    ocp = np.array(ocp)

    with gpflow.defer_build():
        m = gpflow.models.GPR(
            X,
            ocp[:, None],
            kern=gpflow.kernels.RBF(2, ARD=True, variance=1.0)
            + gpflow.kernels.White(2),
            mean_function=gpflow.mean_functions.Constant(),
        )

    m.kern.kernels[0].lengthscales.prior = gpflow.priors.LogNormal(
        [np.log(30), np.log(30)], [1.0, 1.0]
    )
    m.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)

    # make grid predictions
    dx = 0.1
    w = 40
    xx, yy = np.meshgrid(np.arange(-w, w + dx, dx), np.arange(-w, w + dx, dx),)
    h, w = xx.shape
    extent = (np.min(xx), np.max(xx), np.min(xx), np.max(xx))
    gridpoints = np.c_[xx.ravel(), yy.ravel()]

    mu_y, var_y = m.predict_y(gridpoints)

    # no queries closer than 11mm to the edge of the wafer...
    R_max = (76.2 / 2) - 11
    sel = np.sqrt(np.square(xx) + np.square(yy)) > R_max
    var_y[sel.flatten()] = 0

    query_id = np.argmax(var_y)
    query_position = gridpoints[query_id]

    # plot the model...
    figure_path = os.path.join(data_dir, "ocp_predictions_{}.png".format(idx))
    visualization.plot_ocp_model(x, y, ocp, gridpoints, m, query_position, figure_path)

    # convert query position to pd.Series format...
    query_position = pd.Series(dict(x=query_position[0], y=query_position[1]))

    return query_position
