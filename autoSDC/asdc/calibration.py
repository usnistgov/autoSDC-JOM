import os
import re
import datetime
import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate


def composition(solution):
    m = re.match("\d+.\d+", solution)
    try:
        return float(m.group())
    except AttributeError:
        return np.nan


def experiment_start_time(header, verbose=False):
    """ parse times from Jae's excel format """
    # get the time
    t = header.iloc[2, 1].strip()
    dt = datetime.datetime.strptime(t, "%H:%M:%S %p %z")
    if verbose:
        print("time:", dt.time().isoformat())

    # get the date
    t = header.iloc[1, 1:4].values.astype(str)
    t[-1] = int(float(t[-1]))
    t = " ".join(map(str.strip, t))
    _dt = datetime.datetime.strptime(t, "%A %B %d %Y").date()
    if verbose:
        print("date:", _dt.isoformat())

    # combine date and time
    return dt.replace(year=_dt.year, month=_dt.month, day=_dt.day)


def parse_calibration_sheet(df, name=None, verbose=False):
    # pattern match our way through the header to set up extraction
    h = df.iloc[0]
    concentration_pattern = "\d+.\d+ M \w+"
    solution_ids = h.str.match(concentration_pattern, na=False)

    bg_pattern = "boric acid background"
    solution_ids[h == bg_pattern] = True

    s = h[np.where(solution_ids)[0]].to_dict()
    s = [(v, k) for k, v in s.items()]

    # hardcode replicates...
    solutions = []
    replicate_offset = 5
    for key, idx in s:
        solutions.append((key, idx))
        if "M" in key:
            solutions.append((key, idx + replicate_offset))

    # pull all the data out of the excel sheet...
    data = []
    for key, idx in solutions:
        offset = idx + 1
        if verbose:
            print(key, offset)

        row = {"solution": key}

        # load experiment header data
        h = df.iloc[:3, offset : offset + 4]
        row["start_time"] = experiment_start_time(h).isoformat()

        # load results
        d = df.iloc[4:, offset : offset + 3]
        d = d.dropna()
        d.columns = ["time", "voltage", "current"]
        row.update(
            {col: np.array(d.values)[:, idx] for idx, col in enumerate(d.columns)}
        )

        data.append(row)

    data = pd.DataFrame(data)
    data["calibration"] = name
    return data


def load_calibration_data(
    calibration_file="../data/calibration/Ni-Co-calibration-curves-2019-04-23.xlsx",
):
    # Ni and Co data live in different sheets
    # for each, we have three solution concentrations each with two replicates
    # then there's a boric acid background run
    xl = pd.ExcelFile(calibration_file)

    data = []
    for sheet in xl.sheet_names:
        name, *rest = sheet.split(None)
        print(name)
        df = xl.parse(sheet, header=None)
        data.append(parse_calibration_sheet(df, name=name))

    data = pd.concat(data, ignore_index=True)
    data["composition"] = data.solution.apply(composition)

    return data


def load_old_calibration(calibration_file="../data/Nickel and Boric Acid Data.xlsx"):
    # load calibration data from two Excel sheets
    df = pd.read_excel(calibration_file)

    solution = {"0.2 M Ni": 0, "0.1 M Ni": 4, "0.025 M Ni": 8, "Ni": 12, "Co": 16}

    data = []
    for key, offset in solution.items():
        row = {"solution": key}
        row
        d = df.iloc[3:, offset : offset + 3]
        d = d.dropna()
        d.columns = ["time", "voltage", "current"]
        row.update(
            {col: np.array(d.values)[:, idx] for idx, col in enumerate(d.columns)}
        )
        data.append(row)

    data = pd.DataFrame(data)
    data["calibration"] = "Nickel"
    data["calibration"][data["solution"] == "Co"] = "Cobalt"

    df = pd.read_excel("../data/Cobalt Series of Depositions.xlsx")

    solution = {
        "0.1 M Co": 2,
        "0.05 M Co": 6,
        "0.025 M Co": 10,
    }
    data2 = []
    for key, offset in solution.items():
        row = {"solution": key}
        d = df.iloc[:, offset : offset + 3]
        d = d.dropna()
        d.columns = ["voltage", "current", "abscurrent"]
        row.update(
            {col: np.array(d.values)[:, idx] for idx, col in enumerate(d.columns)}
        )
        data2.append(row)

    data2 = pd.DataFrame(data2)
    data2["calibration"] = "Cobalt"

    data = pd.concat([data, data2], ignore_index=True)
    data["solution"][data["solution"].isin(("Ni", "Co"))] = "boric acid background"

    data["composition"] = data.solution.apply(composition)
    return data
