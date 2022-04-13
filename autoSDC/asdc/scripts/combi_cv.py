#!/usr/bin/env python
import os
import sys
import glob
import json
import time
import yaml
import click
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from asdc import sdc
from asdc import visualization

# 5mm up, 50 microns down
@click.command()
@click.argument("config-file", type=click.Path())
@click.option("--verbose/--no-verbose", default=False)
def run_combi_scan(config_file, verbose):
    """keep in mind that sample frame and versastat frame have x and y flipped:
    x_combi is -y_versastat
    y_combi is -x_versastat
    Also, combi wafer frame is in mm, versastat frame is in meters.
    Assume we start at the standard combi layout spot 1 (-9.04, -31.64)
    """

    with open(config_file, "r") as f:
        config = yaml.load(f)
        if config["data_dir"] is None:
            config["data_dir"] = os.path.split(config_file)[0]

    if config["delta_z"] is not None:
        config["delta_z"] = abs(config["delta_z"])

    df = pd.read_csv(config["target_file"], index_col=0)

    current_spot = pd.Series(dict(x=-9.04, y=-31.64))

    with sdc.position.controller(ip="192.168.10.11", speed=config["speed"]) as pos:
        initial_versastat_position = pos.current_position()

    for idx, target in df.iterrows():

        # update position: convert from mm to m
        # x_vs is -y_c, y_vs is x
        dy = -(target.x - current_spot.x) * 1e-3
        dx = -(target.y - current_spot.y) * 1e-3
        delta = [dx, dy, 0.0]
        current_spot = target

        if verbose:
            print("position update:", dx, dy)

        with sdc.position.controller(ip="192.168.10.11", speed=config["speed"]) as pos:
            pos.update(
                delta=delta,
                step_height=config["delta_z"],
                compress=config["compress_dz"],
            )
            current_v_position = pos.current_position()

        # run CV scan
        if config["initial_delay"]:
            time.sleep(config["initial_delay"])
        cv_data = sdc.experiment.run_cv_scan(
            cell=config["cell"], verbose=config["verbose"]
        )
        cv_data["index_in_sequence"] = int(idx)
        cv_data["position_versa"] = current_v_position
        _spot = current_spot.to_dict()
        cv_data["position_combi"] = [float(_spot["x"]), float(_spot["y"])]

        # log data
        logfile = "grid_scan_{:03d}.json".format(idx)
        with open(os.path.join(config["data_dir"], logfile), "w") as f:
            json.dump(cv_data, f)

        visualization.plot_iv(
            cv_data["current"], cv_data["potential"], idx, config["data_dir"]
        )
        visualization.plot_v(
            cv_data["elapsed_time"],
            cv_data["potential"],
            idx,
            data_dir=config["data_dir"],
        )

    # go back to the original position....
    with sdc.position.controller(ip="192.168.10.11", speed=config["speed"]) as pos:
        x_initial, y_initial, z_initial = initial_versastat_position
        x_current, y_current, z_current = pos.current_position()
        delta = [x_initial - x_current, y_initial - y_current, 0.0]

        pos.update(
            delta=delta, step_height=config["delta_z"], compress=config["compress_dz"]
        )


if __name__ == "__main__":
    run_combi_scan()
