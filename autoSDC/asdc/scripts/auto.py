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
from asdc import ocp
from asdc import slack
from asdc import analyze
from asdc import visualization


def update_position_combi(
    target, current_spot, speed=1e-3, delta_z=5e-3, compress_dz=5e-5, verbose=False
):
    # update position: convert from mm to m
    # x_vs is -y_c, y_vs is x
    dy = -(target.x - current_spot.x) * 1e-3
    dx = -(target.y - current_spot.y) * 1e-3
    delta = [dx, dy, 0.0]
    current_spot = target

    if verbose:
        print("position update:", dx, dy)

    with sdc.position.controller(ip="192.168.10.11", speed=speed) as pos:

        pos.update(delta=delta, step_height=delta_z, compress=compress_dz)
        current_v_position = pos.current_position()

    return current_v_position


# 5mm up, 50 microns down
@click.command()
@click.argument("config-file", type=click.Path())
@click.option("--verbose/--no-verbose", default=False)
def run_auto_scan(config_file, verbose):
    """collect three CV curves from predetermined spots, then use a Gaussian Process model to map out open circuit potential

    keep in mind that sample frame and versastat frame have x and y flipped:
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
        config["delta_z"] = np.clip(abs(config["delta_z"]), 0, 1e-1)

    # assume we start at standard combi layout spot 1
    current_spot = pd.Series(dict(x=-9.04, y=-31.64))

    # get the corresponding versastat reference coordinates
    with sdc.position.controller(ip="192.168.10.11", speed=config["speed"]) as pos:
        initial_versastat_position = pos.current_position()

    # kickstart with a few pre-determined scans...
    df = pd.read_csv(config["target_file"], index_col=0)
    n_initial, _ = df.shape
    n_total = n_initial + config["n_acquisitions"]

    pre_collected_data = glob.glob(os.path.join(config["data_dir"], "*.json"))
    # start_idx = len(pre_collected_data)
    most_recent_file = sorted(pre_collected_data)[-1]
    bn, _ = os.path.splitext(os.path.basename(most_recent_file))
    _, _, start_idx = bn.split("_")
    start_idx = int(start_idx)

    for idx in range(start_idx, n_total):

        if idx < n_initial:
            slack.post_message("acquiring predetermined spot {}".format(idx))
            target = df.iloc[idx]
        else:
            slack.post_message("acquiring GP spot {}".format(idx))
            target = ocp.gp_select(config["data_dir"], plot_model=True, idx=idx)
            figpath = os.path.join(
                config["data_dir"], "ocp_predictions_{}.png".format(idx)
            )
            slack.post_image(figpath, title="OCP map {}".format(idx))

        # update position
        # specify target position in combi sample coordinates
        current_v_position = update_position_combi(
            target,
            current_spot,
            delta_z=config["delta_z"],
            compress=config["compress_dz"],
            speed=config["speed"],
        )
        current_spot = target

        # run CV scan
        if config["initial_delay"]:
            time.sleep(config["initial_delay"])
        cv_data = sdc.experiment.run_cv_scan(cell=config["cell"], verbose=verbose)
        cv_data["index_in_sequence"] = int(idx)
        cv_data["position_versa"] = current_v_position
        _spot = current_spot.to_dict()
        cv_data["position_combi"] = [float(_spot["x"]), float(_spot["y"])]

        # log data
        logfile = "cv_scan_{:03d}.json".format(idx)
        with open(os.path.join(config["data_dir"], logfile), "w") as f:
            json.dump(cv_data, f)

        figpath = os.path.join(config["data_dir"], "open_circuit_{}.png".format(idx))
        visualization.plot_open_circuit(
            cv_data["current"],
            cv_data["potential"],
            cv_data["segment"],
            figpath=figpath,
        )
        slack.post_image(figpath, title="open circuit {}".format(idx))

        # visualization.plot_iv(cv_data['current'], cv_data['potential'], idx, data_dir)
        # visualization.plot_v(cv_data['elapsed_time'], cv_data['potential'], idx, data_dir=data_dir)

    # re-fit the GP after the final measurement
    slack.post_message("fitting final GP model.")
    target = ocp.gp_select(config["data_dir"], plot_model=True)
    slack.post_image(
        os.path.join(config["data_dir"], "ocp_predictions_final.png"), title="OCP map"
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
    run_auto_scan()
