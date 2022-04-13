import os
import json
import dataset
import pandas as pd
import numpy as np
from asdc import analysis


def process_pH(pH_file, dir="data"):
    data = {}
    try:
        df = pd.read_csv(os.path.join(dir, pH_file))
        data["pH_initial"] = df.pH.iloc[0]
        data["pH_final"] = df.pH.iloc[-1]
        data["pH_avg"] = df.pH.mean()
        data["pH_med"] = df.pH.median()
        data["T_initial"] = df.temperature.iloc[0]
        data["T_avg"] = df.temperature.mean()
    except:
        pass

    return data


def process_ocp(experiments, dir="data"):
    data = {}

    # there should be only one!
    experiment = experiments[0]
    ocp = analysis.OCPData(pd.read_csv(os.path.join(dir, experiment["datafile"])))

    data["E_oc_initial"] = ocp.potential.iloc[0]
    data["E_oc_stable"] = ocp.potential.iloc[-20:].mean()
    data["E_oc_stable_std"] = ocp.potential.iloc[-20:].std()

    return data


def process_lpr(experiments, dir="data", r2_thresh=0.95):
    data = []

    for experiment in experiments:
        datafile = os.path.join(dir, experiment["datafile"])
        lpr = pd.read_csv(datafile)
        lpr = analysis.LPRData(lpr)
        lpr.check_quality(r2_thresh=r2_thresh)

        pr, ocp, r2 = lpr.fit()
        if r2 < r2_thresh:
            pr = np.nan
            ocp = np.nan
            r2 = np.nan
        data.append({"polarization_resistance": pr, "E_oc": ocp, "lpr_r2": r2})

    df = pd.DataFrame(data)

    data = {}
    data["polarization_resistance"] = np.nanmean(df["polarization_resistance"])
    data["polarization_std"] = np.nanstd(df["polarization_resistance"])
    data["lpr_r2"] = np.nanmean(df["lpr_r2"])
    data["lpr_r2_std"] = np.nanstd(df["lpr_r2"])
    data["lpr_E_oc"] = np.nanmean(df["E_oc"])
    data["lpr_E_oc_std"] = np.nanstd(df["E_oc"])

    return data


def process_tafel(experiment, dir="data"):

    tafel = analysis.TafelData(pd.read_csv(os.path.join(dir, experiment["datafile"])))
    tafel.clip_current_to_range()

    model = tafel.fit()
    return {"tafel_E_oc": tafel.ocp, "i_corr": tafel.i_corr}


def process_row(row, db, dir=dir):

    loc_id = row["id"]

    instructions = json.loads(row["instructions"])

    # unpack flowrate stuff
    flow_instructions = instructions[0]
    row["flow_rate"] = flow_instructions["flow_rate"]
    row["relative_rates"] = flow_instructions["relative_rates"]
    row["purge_time"] = flow_instructions["purge_time"]

    # read and process pH logs
    # oops -- forgot to save pH logfile in the db...
    pH_file = f"pH_log_run{loc_id:03d}.csv"
    pH_metadata = process_pH(pH_file, dir=dir)
    row.update(pH_metadata)

    # analyze OCP trace
    ocp_experiments = list(db["experiment"].find(location_id=loc_id, op="open_circuit"))
    if len(ocp_experiments) > 0:
        ocp_metadata = process_ocp(ocp_experiments, dir=dir)
        row.update(ocp_metadata)

    # analyze LPR data
    lpr_experiments = list(db["experiment"].find(location_id=loc_id, op="lpr"))
    if len(lpr_experiments) > 0:
        lpr_metadata = process_lpr(lpr_experiments, dir=dir)
        row.update(lpr_metadata)

    # analyze Tafel data
    tafel_experiment = db["experiment"].find_one(location_id=loc_id, op="tafel")
    if tafel_experiment:
        tafel_metadata = process_tafel(tafel_experiment, dir=dir)
        row.update(tafel_metadata)

    # analyze CV data

    return row


def load_session(db_path: str, verbose: bool = False) -> pd.DataFrame:
    """load metadata and analyze results

    scan number (scan numbers by type (this will be useful if we begin more dynamic scan orders))
    location
    flowrates
    pH (median?)
    beginning and final ocp from hold
    outputs from LPR fit
    outputs from Tafel fit
    start time
    """

    dir, _ = os.path.split(db_path)
    records = []
    with dataset.connect(f"sqlite:///{db_path}") as db:
        for location in db["location"]:
            if verbose:
                print(location)
            records.append(process_row(location, db, dir=dir))

    return pd.DataFrame(records)
