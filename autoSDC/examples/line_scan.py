#!/usr/bin/env python

import json
import time
from datetime import datetime

import os
import sys

import asdc.position
import asdc.control


def line_scan(speed=1e-5, poll_interval=5):
    """ perform a line scan with CV experiments, recording position, current, potential, and parameters in json log files
    Position units are METERS!
    """

    delta = [1e-4, 1e-4, 0.0]
    initial_delta = [0.0, 0.0, -4.90e-4]
    final_delta = [0.0, 0.0, 4.90e-4]
    n_steps = 10

    with asdc.position.controller(ip="192.168.10.11", speed=speed) as pos:

        pos.print_status()
        pos.update(delta=initial_delta, verbose=True)
        pos.print_status()

        with asdc.control.controller(start_idx=17109013) as pstat:
            pstat.set_current_range("20nA")

            for idx in range(n_steps):
                # scan, log, take a position step

                # run a CV experiment
                status, params = pstat.multi_cyclic_voltammetry(
                    initial_potential=0.0,
                    vertex_potential_1=-0.25,
                    vertex_potential_2=0.65,
                    final_potential=0.0,
                    scan_rate=0.2,
                    cell_to_use="EXTERNAL",
                    e_filter="1Hz",
                    i_filter="1Hz",
                )

                pstat.start()

                while pstat.sequence_running():
                    time.sleep(poll_interval)

                # collect and log data
                scan_data = {
                    "measurement": "cyclic_voltammetry",
                    "parameters": params,
                    "index_in_sequence": idx,
                    "timestamp": datetime.now().isoformat(),
                    "current": pstat.current(),
                    "potential": pstat.potential(),
                    "position": pos.current_position(),
                }

                logfile = "line_scan_{:03d}.json".format(idx)
                with open(logfile, "w") as f:
                    json.dump(scan_data, f)

                pstat.clear()

                # update position
                pos.update(delta=delta, verbose=True)
                pos.print_status()

            # bring the probe back up
            pos.update(delta=final_delta, verbose=True)
            pos.print_status()


if __name__ == "__main__":
    line_scan()
