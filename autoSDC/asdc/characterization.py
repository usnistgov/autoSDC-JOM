import json
import dataset
import imageio
import pathlib
import numpy as np
import pandas as pd
from scipy import integrate
from sklearn import linear_model


def load_rates(expt):
    instructions = json.loads(expt["instructions"])
    return instructions[0].get("rates")


def deposition_potential(expt):
    instructions = json.loads(expt["instructions"])
    return instructions[1].get("potential")


def load_image(expt, data_dir="data"):
    data_dir = pathlib.Path(data_dir)
    return imageio.imread(data_dir / expt["image_name"])


def load_deposition_data(expt, data_dir="data"):
    data_dir = pathlib.Path(data_dir)
    return pd.read_csv(data_dir / expt["datafile"], index_col=0)


def load_corrosion_data(experiment_table, expt, data_dir="data"):
    data_dir = pathlib.Path(data_dir)
    first_rep = experiment_table.find_one(experiment_id=expt["experiment_id"])
    if expt["id"] != first_rep["id"]:
        return None
    corr = experiment_table.find_one(
        experiment_id=expt["experiment_id"], intent="corrosion"
    )
    if corr is not None:
        return pd.read_csv(data_dir / corr["datafile"], index_col=0)


def polarization_resistance(expt, experiment_table, data_dir="data"):
    """ compute polarization resistance (Ohms) """

    data_dir = pathlib.Path(data_dir)
    corr = load_corrosion_data(experiment_table, expt, data_dir=data_dir)

    if corr is None:
        return np.nan

    pr = corr[corr["segment"] == 0]

    log_I = np.log10(np.abs(pr["current"]))
    idx = log_I.idxmin()
    n_skip = int(pr.shape[0] * 0.3)
    slc = slice(idx - n_skip, idx + n_skip)

    lm = linear_model.HuberRegressor()

    try:
        lm.fit(pr["potential"][slc, None], pr["current"][slc])
        return 1 / lm.coef_[0]

    except ValueError:
        return np.nan


def load_laser_data(expt, data_dir="data"):
    data_dir = pathlib.Path(data_dir)

    if expt["reflectance_file"] is None:
        # print(expt['id'])
        return [np.nan]

    with open(data_dir / expt["reflectance_file"], "r") as f:
        data = json.load(f)

    return data["reflectance"]


def load_xrf_file(datafile, header_rows=32):

    # load the column names
    with open(datafile, "r") as f:
        lines = f.readlines()

    header = lines[header_rows]
    names = header.strip()[2:].split()

    return pd.read_csv(
        datafile, skiprows=header_rows + 1, delim_whitespace=True, names=names
    )


def load_xrf_data(expt, data_dir="data", scan="middle"):
    id = expt["id"]

    if scan not in ("down", "middle", "up", "slits"):
        raise ValueError("scans should be one of {down, middle, up}")

    data_dir = pathlib.Path(data_dir)

    if scan == "slits":
        # switch to sdc-25-{id:04d}_slitscan.dat
        datafile = data_dir / "xray" / f"sdc-26-{id:04d}_slitscan.dat"
    else:
        datafile = data_dir / "xray" / f"sdc-26-{id:04d}_linescan_{scan}.dat"

    # print(datafile)
    try:
        return load_xrf_file(datafile)
    except FileNotFoundError:
        return None


def load_reference_xrf(reference_datafile="rawgold.dat", data_dir="data"):

    data_dir = pathlib.Path(data_dir)

    reference = load_xrf_file(data_dir / "xray" / reference_datafile, header_rows=17)
    reference = reference / reference["I0"][:, None]

    Au_counts = np.median(reference["DTC3_1"])
    Ni_bg = Au_counts / np.median(reference["DTC1"])
    Zn_bg = Au_counts / np.median(reference["DTC2_1"])

    return {"Ni": Ni_bg, "Zn": Zn_bg}


def xrf_Ni_ratio(expt, midpoint=False, data_dir="data", scan="middle"):
    """
    Dead-time-corrected counts:
    Ni: DTC1  Zn: DTC2_1  Au: DTC3_1
    Au vs elastic: DTC3_2
    Ni: ROI1
    Zn: ROI2_1
    Au: ROI3_1
    """
    data_dir = pathlib.Path(data_dir)
    # print(data_dir)
    xrf = load_xrf_data(expt, data_dir=data_dir, scan=scan)

    # print(xrf)

    if xrf is None:
        return [np.nan]

    xrf = xrf / xrf["I0"][:, None]

    bg = load_reference_xrf(data_dir=data_dir)

    Ni_counts = xrf["DTC1"] - bg["Ni"]
    Zn_counts = xrf["DTC2_1"] - bg["Zn"]

    NiZn_counts = Ni_counts + Zn_counts

    Ni = Ni_counts / NiZn_counts

    if midpoint:
        midpoint_idx = Ni.size // 2
        return Ni[30]

    return Ni


def integral_corrosion_current(expt, experiment_table, start_time=5, data_dir="data"):

    data_dir = pathlib.Path(data_dir)
    corr = load_corrosion_data(experiment_table, expt, data_dir=data_dir)

    if corr is None:
        return np.nan

    if len(corr["segment"].unique()) > 1:
        corr = corr[corr["segment"] == 1]

    s = corr["elapsed_time"] > start_time
    integral_current = integrate.trapz(corr["current"][s], corr["elapsed_time"][s])
    return integral_current


def load_results(expt, experiment_table, data_dir, scan="middle"):

    rates = load_rates(expt)
    res = {
        "refl": np.mean(load_laser_data(expt, data_dir=data_dir)),
        # 'Ni_ratio': xrf_Ni_ratio(expt, midpoint=True),
        "potential": deposition_potential(expt),
        "Ni_ratio": np.median(xrf_Ni_ratio(expt, data_dir=data_dir, scan=scan)[7:14]),
        "Ni_variance": np.var(xrf_Ni_ratio(expt, data_dir=data_dir, scan=scan)[3:18]),
        "integral_current": integral_corrosion_current(
            expt, experiment_table, data_dir=data_dir
        ),
        "polarization_resistance": polarization_resistance(
            expt, experiment_table, data_dir=data_dir
        ),
    }
    res.update(rates)
    res["id"] = expt["experiment_id"]
    return res


def load_characterization_results(dbfile, scan="slits"):

    print(scan)
    dbfile = pathlib.Path(dbfile)
    data_dir = dbfile.parent

    db = dataset.connect(f"sqlite:///{str(dbfile)}")
    experiment_table = db["experiment"]

    df = pd.DataFrame(
        [
            load_results(e, experiment_table, data_dir, scan=scan)
            for e in experiment_table.find(intent="deposition")
        ]
    )
    return df
