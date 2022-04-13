import os
import glob
import json
import gpflow
import dataset
import numpy as np
import pandas as pd
from datetime import datetime

from asdc import analyze

flags = [36, 38]
elements = ["Ni", "Al", "Ti"]


def model_composition(df, invert_x=True, convert_to_decimal=True):

    X = df.loc[:, ("0", "1")].values.astype(np.float)

    # wafer and composition measurement reference frames
    # are mirrored in x
    if invert_x:
        X[:, 0] *= -1

    Y = df.loc[:, elements].values
    if convert_to_decimal:
        Y *= 0.01

    N, D = X.shape

    m = gpflow.models.GPR(
        X, Y, kern=gpflow.kernels.RBF(2), mean_function=gpflow.mean_functions.Constant()
    )

    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m, disp=True)
    return m


def model_fwhm(df, invert_x=True):

    X = df.loc[:, ("0", "1")].values.astype(np.float)

    # wafer and composition measurement reference frames
    # are mirrored in x
    if invert_x:
        X[:, 0] *= -1

    F = df["FWHM"].values.astype(np.float)[:, None]

    N, D = X.shape

    m = gpflow.models.GPR(
        X,
        F,
        kern=gpflow.kernels.Matern32(2),
        mean_function=gpflow.mean_functions.Constant(),
    )

    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m, disp=True)
    return m


def extract_cv_features(data, start=1120):
    # first find the right segment of the data...
    I = np.array(data["current"])[start:]
    V = np.array(data["potential"])[start:]

    # now correct for autorange artifacts
    a = analyze.model_autorange_artifacts(V, I, tau_increasing=10)
    log_I = np.log10(np.abs(I)) - a

    # extract features from polarization curve
    log_I, m, lm, cv_features = analyze.model_polarization_curve(
        V, log_I, bg_order=7, lm_method="huber", smooth=True
    )

    cv_features["slope"] = lm.coef_[0]

    return cv_features


def analyze_NiTiAl():
    datafiles = sorted(glob.glob("data/NiTiAl-K20/*.json"))
    db_file = "data/k20-NiTiAl.db"
    db = dataset.connect(f"sqlite:///{db_file}")

    invert_x = True
    convert_to_decimal = True
    k20 = pd.read_csv("data/k20.csv", index_col=0)
    composition = model_composition(
        k20, invert_x=invert_x, convert_to_decimal=convert_to_decimal
    )
    fwhm = model_fwhm(k20, invert_x=invert_x)

    for idx, datafile in enumerate(datafiles):
        with open(datafile, "r") as ff:
            d = json.load(ff)

        d["flag"] = True if d["index_in_sequence"] in flags else False

        d["timestamp_start"] = datetime.fromisoformat(d["timestamp_start"][0])
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])

        if not d["flag"]:
            cv_features = extract_cv_features(d)
            # convert to native types
            cv_features = {key: float(val) for key, val in cv_features.items()}
            d.update(cv_features)

        # unpack positions
        d["x_combi"] = d["position_combi"][0]
        d["y_combi"] = d["position_combi"][1]
        d["x_versa"] = d["position_versa"][0]
        d["y_versa"] = d["position_versa"][1]
        d["z_versa"] = d["position_versa"][2]

        X_query = np.array((d["x_combi"], d["y_combi"]))[None, :]
        C, C_var = composition.predict_y(X_query)
        for el, comp in zip(elements, C.flat):
            d[el] = comp

        F, F_var = fwhm.predict_y(X_query)
        d["fwhm"] = float(F)

        # pack results into json string
        res = {
            "current": d["current"],
            "potential": d["potential"],
            "error_codes": d["error_codes"],
        }
        d["result"] = json.dumps(res)

        for key in res.keys():
            del d[key]

        del d["position_combi"]
        del d["position_versa"]

        with db as tx:
            db["experiment"].insert(d)


if __name__ == "__main__":
    analyze_NiTiAl()
