#!/usr/bin/env python
import os
import sys
import glob
import json
import time
import click
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from asdc import sdc
from asdc import visualization


@click.group()
def cli():
    pass


@cli.command()
@click.option("--verbose/--no-verbose", default=False)
def reset(verbose):
    """ try to reset the potentiostat controller... """
    with sdc.potentiostat.controller(start_idx=17109013) as pstat:
        pstat.stop()
        pstat.clear()


@cli.command()
@click.option(
    "-d",
    "--direction",
    default="x",
    type=click.Choice(["x", "y", "+x", "-x", "+y", "-y"]),
)
@click.option("--delta", default=5e-3, type=float, help="x step in meters")
@click.option("--delta-z", default=5e-5, type=float, help="z step in meters")
@click.option("--speed", default=1e-3, type=float, help="speed in meters/s")
@click.option(
    "--lift/--no-lift",
    default=False,
    help="ease off vertically before horizontal motion.",
)
@click.option(
    "--press/--no-press",
    default=True,
    help="press down below vertical setpoint to reseat probe after horizontal motion",
)
@click.option("--verbose/--no-verbose", default=False)
def step(direction, delta, delta_z, speed, lift, press, verbose):
    """1mm per second scan speed.
    up. over. down. down. up
    lift: up, over, down.
    press: after horizontal step, press down and release by delta_z
    """

    # constrain absolute delta_z to avoid crashing....
    delta_z = np.clip(delta_z, -5e-5, 5e-5)

    with sdc.position.controller(ip="192.168.10.11", speed=speed) as pos:

        # vertical step
        if lift:
            pos.update_z(delta=delta_z, verbose=verbose)

        # take the position step
        if verbose:
            pos.print_status()

        if "x" in direction:
            update_position = pos.update_x
        elif "y" in direction:
            update_position = pos.update_y

        if "-" in direction:
            delta *= -1

        update_position(delta=delta, verbose=verbose)

        if verbose:
            pos.print_status()

        if lift:
            # vertical step back down:
            pos.update_z(delta=-delta_z, verbose=verbose)

        if press:
            # compress, then release
            pos.update_z(delta=-delta_z, verbose=verbose)
            pos.update_z(delta=delta_z, verbose=verbose)

        if verbose:
            pos.print_status()


@cli.command()
@click.option("--data-dir", default="data", type=click.Path())
@click.option(
    "-c", "--cell", default="INTERNAL", type=click.Choice(["INTERNAL", "EXTERNAL"])
)
@click.option("--verbose/--no-verbose", default=False)
def cv(data_dir, cell, verbose):
    """ run a CV experiment """

    # check on experiment status periodically:
    poll_interval = 1

    # load previous datasets just to get current index...
    datafiles = glob.glob(os.path.join(data_dir, "*.json"))
    scan_idx = len(datafiles)
    with sdc.position.controller(ip="192.168.10.11") as pos:

        cv_data = sdc.experiment.run_cv_scan(cell=cell, verbose=verbose)

        logfile = "cv_{:03d}.json".format(scan_idx)
        with open(os.path.join(data_dir, logfile), "w") as f:
            json.dump(cv_data, f)

        print("plotting...")
        visualization.plot_v(
            cv_data["elapsed_time"], cv_data["potential"], scan_idx, data_dir=data_dir
        )
        print("first plot done")
        visualization.plot_iv(
            cv_data["current"], cv_data["potential"], scan_idx, data_dir
        )
        print("second plot done")

    return


if __name__ == "__main__":
    cli()
