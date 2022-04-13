import os
import sys
import json
import time
import logging
import argparse
import asyncio
import dataset
import functools
import subprocess

# import websockets
import numpy as np
import pandas as pd
from ruamel import yaml
from datetime import datetime
from aioconsole import ainput, aprint
from contextlib import contextmanager, asynccontextmanager

from typing import Any, List, Dict, Optional, Tuple, Union

import traceback

import cv2
import imageio

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sympy
from sympy import geometry
from sympy.vector import express
from sympy.vector import CoordSys3D, BodyOrienter, Point

sys.path.append(".")

from asdc import sdc
from asdc import epics
from asdc import _slack
from asdc import slackbot
from asdc import visualization
from asdc.analysis import EchemData

from asdc.sdc.reglo import Channel

potentiostat_id = 17109013

asdc_channel = "CDW5JFZAR"
try:
    BOT_TOKEN = open("slacktoken.txt", "r").read().strip()
except FileNotFoundError:
    BOT_TOKEN = None

try:
    CTL_TOKEN = open("slack_bot_token.txt", "r").read().strip()
except FileNotFoundError:
    CTL_TOKEN = None

# reference to web client...
web_client = _slack.sc

# logger = logging.getLogger(__name__)
logger = logging.getLogger()


def save_plot(results: EchemData, figpath: str, post_slack: bool = True, title=None):

    try:
        results.plot()
    except Exception as err:
        logger.error(f"data check: {err}")

    plt.savefig(figpath, bbox_inches="tight")
    plt.clf()
    plt.close()
    if post_slack:
        _slack.post_image(web_client, figpath, title=title)


def relative_flow(rates):
    """ convert a dictionary of flow rates to ratios of each component """
    total = sum(rates.values())
    if total == 0.0:
        return rates
    return {key: rate / total for key, rate in rates.items()}


def to_vec(x, frame):
    """ convert python iterable coordinates to vector in specified reference frame """
    return x[0] * frame.i + x[1] * frame.j


def to_coords(x, frame):
    """ express coordinates in specified reference frame """
    return frame.origin.locate_new("P", to_vec(x, frame))


class SDC:
    """ scanning droplet cell """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        token: str = BOT_TOKEN,
        resume: bool = False,
        logfile: Optional[str] = None,
        verbose: bool = False,
        zmq_pub: bool = False,
    ):
        """scanning droplet cell client

        this is a slack client that controls all of the hardware and executes experiments.

        Arguments:
            config: configuration dictionary
            token: slack bot token
            resume: toggle auto-registration of stage and sample coordinates
            logfile: file to log slackbot commands to
            verbose: toggle additional debugging output

        """
        self.verbose = verbose
        self.logfile = logfile
        self.configvalues = config

        with sdc.position.controller(ip="192.168.10.11") as pos:
            initial_versastat_position = pos.current_position()
            logger.debug(f"initial vs position: {initial_versastat_position}")

        self.initial_versastat_position = initial_versastat_position
        self.initial_combi_position = pd.Series(config["initial_combi_position"])
        self.step_height = config.get("step_height", 0.0)
        self.cleanup_pause = config.get("cleanup_pause", 0)
        self.cleanup_pulse_duration = config.get("cleanup_pulse_duration", 0)
        self.cell = config.get("cell", "INTERNAL")
        self.speed = config.get("speed", 1e-3)
        self.data_dir = config.get("data_dir", os.getcwd())
        self.figure_dir = config.get("figure_dir", os.getcwd())
        self.confirm = config.get("confirm", True)
        self.confirm_experiment = config.get("confirm_experiment", True)
        self.notify = config.get("notify_slack", True)
        self.plot_slack = config.get("plot_slack", False)
        self.plot_cv = config.get("plot_cv", False)
        self.plot_current = config.get("plot_current", False)

        self.default_experiment = config.get("default_experiment")
        if self.default_experiment is not None and type(self.default_experiment) is str:
            self.default_experiment = json.loads(self.default_experiment)

        self.default_flowrate = config.get("default_flowrate")

        # define a positive height to perform characterization
        h = float(config.get("characterization_height", 0.004))
        h = max(0.0, h)
        self.characterization_height = h

        # define a positive height to perform characterization
        h = float(config.get("laser_scan_height", 0.015))
        h = max(0.0, h)
        self.laser_scan_height = h
        self.xrays_height = self.laser_scan_height

        self.camera_index = int(config.get("camera_index", 2))

        # droplet workflow configuration
        # TODO: document me
        self.wetting_height = max(0, config.get("wetting_height", 0.0011))
        self.droplet_height = max(0, config.get("droplet_height", 0.004))
        self.fill_rate = config.get("fill_rate", 1.0)
        self.fill_counter_ratio = config.get("fill_counter_ratio", 0.7)
        self.fill_time = config.get("fill_time", 19)
        self.shrink_counter_ratio = config.get("shrink_counter_ratio", 1.3)
        self.shrink_time = config.get("shrink_time", 2)

        self.test = config.get("test", False)
        self.test_cell = config.get("test_cell", False)
        self.solutions = config.get("solutions")

        self.v_position = self.initial_versastat_position
        self.c_position = self.initial_combi_position

        self.initialize_z_position = config.get("initialize_z_position", False)

        # which wafer direction is aligned with position controller +x direction?
        self.frame_orientation = config.get("frame_orientation", "-y")

        self.db_file = os.path.join(self.data_dir, config.get("db_file", "testb.db"))
        self.db = dataset.connect(f"sqlite:///{self.db_file}")
        self.location_table = self.db["location"]
        self.experiment_table = self.db["experiment"]

        self.current_threshold = 1e-5

        self.resume = resume

        # define reference frames
        # load camera and laser offsets from configuration file
        camera_offset = config.get("camera_offset", [38.3, -0.4])
        laser_offset = config.get("laser_offset", [38, -0.3])
        xray_offset = config.get("xray_offset", [44.74, -4.4035])

        self.cell_frame = CoordSys3D("cell")
        self.camera_frame = self.cell_frame.locate_new(
            "camera",
            camera_offset[0] * self.cell_frame.i + camera_offset[1] * self.cell_frame.j,
        )
        self.laser_frame = self.cell_frame.locate_new(
            "laser",
            laser_offset[0] * self.cell_frame.i + laser_offset[1] * self.cell_frame.j,
        )
        self.xray_frame = self.cell_frame.locate_new(
            "xray",
            xray_offset[0] * self.cell_frame.i + xray_offset[1] * self.cell_frame.j,
        )

        if self.resume:
            self.stage_frame = self.sync_coordinate_systems(
                orientation=self.frame_orientation,
                register_initial=True,
                resume=self.resume,
            )
        else:
            self.stage_frame = self.sync_coordinate_systems(
                orientation=self.frame_orientation, register_initial=False
            )

        reglo_port = config.get("reglo_port", "COM16")
        orion_port = config.get("orion_port", "COM17")
        adafruit_port = config.get("adafruit_port", "COM9")
        pump_array_port = config.get("pump_array_port", "COM10")
        self.backfill_duration = config.get("backfill_duration", 15)

        diameter = config.get("syringe_diameter", 29.5)

        try:
            self.pump_array = sdc.pump.PumpArray(
                self.solutions, port=pump_array_port, timeout=1, diameter=diameter
            )
        except:
            logger.exception("could not connect to pump array")
            # raise
            self.pump_array = None

        try:
            self.reglo = sdc.reglo.Reglo(
                address=reglo_port, debug=config.get("reglo_debug", False)
            )
        except:
            logger.exception("could not connect to the Reglo peristaltic pump")
            # raise

        try:
            self.phmeter = sdc.orion.PHMeter(orion_port, zmq_pub=zmq_pub)
        except:
            logger.exception("could not connect to the Orion pH meter")
            # raise

        try:
            self.reflectometer = sdc.microcontroller.Reflectometer(port=adafruit_port)
            self.light = sdc.microcontroller.Light(port=adafruit_port)
        except:
            logger.exception("could not connect to the adafruit board")
            self.reflectometer = None
            self.light = None

    def get_last_known_position(self, x_versa, y_versa, resume=False):
        """set up initial cell reference relative to a previous database entry if possible

        If not, or if `resume` is False, set initial cell reference from config file. It is
        the operator's responsibility to ensure this initial position matches the physical configuration
        """

        # load last known combi position and update internal state accordingly
        refs = pd.DataFrame(self.location_table.all())

        if (resume == False) or (refs.size == 0):

            init = self.initial_combi_position
            logger.info(f"starting from {init}")

            ref = pd.Series(
                {
                    "x_versa": x_versa,
                    "y_versa": y_versa,
                    "x_combi": init.x,
                    "y_combi": init.y,
                }
            )
        else:
            # arbitrarily grab the first position
            # TODO: verify that this record comes from the current session...
            ref = refs.iloc[0].to_dict()
            ref["x_versa"] *= 1e3
            ref["y_versa"] *= 1e3
            ref = pd.Series(ref)
            logger.info(f"resuming from {ref}")

        return ref

    def current_versa_xy(self):
        """ get current stage coords in mm """

        with sdc.position.controller() as pos:
            x_versa = pos.x * 1e3
            y_versa = pos.y * 1e3

        return x_versa, y_versa

    def locate_wafer_center(self):
        """align reference frames to wafer center

        identify a circumcircle corresponding three points on the wafer edge
        """
        wafer_edge_coords = []
        logger.info(
            "identify coordinates of three points on the wafer edge. (start with the flat corners)"
        )

        for idx in range(3):
            input("press enter to register coordinates...")
            wafer_edge_coords.append(self.current_versa_xy())

        # unpack triangle coordinates
        tri = geometry.Triangle(*wafer_edge_coords)

        # center is the versascan coordinate such that the camera frame is on the wafer origin
        center = np.array(tri.circumcenter, dtype=float)

        logger.debug(f"wafer edge coordinates: {wafer_edge_coords}")
        logger.debug(f"center coordinate: {center}")

        # move the stage to focus the camera on the center of the wafer...
        current = np.array(self.current_versa_xy())
        delta = center - current

        # convert to meters!
        delta = delta * 1e-3
        logger.debug(f"moving cell to center: {delta}")

        # specify updates in the stage frame...
        with sdc.position.controller(speed=self.speed) as stage:
            input("press enter to allow lateral cell motion...")
            stage.update(delta=delta)

        # set up the stage reference frame
        # relative to the last recorded positions
        cam = self.camera_frame

        if self.frame_orientation == "-y":
            _stage = cam.orient_new(
                "_stage", BodyOrienter(sympy.pi / 2, sympy.pi, 0, "ZYZ")
            )
        else:
            raise NotImplementedError

        # find the origin of the combi wafer in the coincident stage frame
        v = 0.0 * cam.i + 0.0 * cam.j
        combi_origin = v.to_matrix(_stage)

        # truncate to 2D vector
        combi_origin = np.array(combi_origin).squeeze()[:-1]

        # now find the origin of the stage frame
        # xv_init = np.array([ref['x_versa'], ref['y_versa']])
        xv_init = np.array(center)

        l = xv_init - combi_origin
        v_origin = l[1] * cam.i + l[0] * cam.j

        # construct the shifted stage frame
        stage = _stage.locate_new("stage", v_origin)
        self.stage_frame = stage

    def sync_coordinate_systems(
        self, orientation=None, register_initial=False, resume=False
    ):
        """ set up stage reference frames relative to the cell coordinate system """

        with sdc.position.controller() as pos:
            # map m -> mm
            x_versa = pos.x * 1e3
            y_versa = pos.y * 1e3

        ref = self.get_last_known_position(x_versa, y_versa, resume=resume)

        # set up the stage reference frame
        # relative to the last recorded positions
        cell = self.cell_frame

        if orientation == "-y":
            _stage = cell.orient_new(
                "_stage", BodyOrienter(sympy.pi / 2, sympy.pi, 0, "ZYZ")
            )
        else:
            raise NotImplementedError

        # find the origin of the combi wafer in the coincident stage frame
        v = ref["x_combi"] * cell.i + ref["y_combi"] * cell.j
        combi_origin = v.to_matrix(_stage)

        # truncate to 2D vector
        combi_origin = np.array(combi_origin).squeeze()[:-1]

        # now find the origin of the stage frame
        xv_init = np.array([ref["x_versa"], ref["y_versa"]])
        if resume:
            offset = np.array([x_versa, y_versa]) - xv_init
            logger.debug(f"wafer offset: {offset}")
            # xv_init += offset

        l = xv_init - combi_origin
        v_origin = l[1] * cell.i + l[0] * cell.j

        # construct the shifted stage frame
        stage = _stage.locate_new("stage", v_origin)
        return stage

    def compute_position_update(self, x: float, y: float, frame: Any) -> np.ndarray:
        """compute frame update to map combi coordinate to the specified reference frame

        Arguments:
            x: wafer x coordinate (`mm`)
            y: wafer y coordinate (`mm`)
            frame: target reference frame (`cell`, `camera`, `laser`)

        Returns:
            stage frame update vector (in meters)

        Important:
            all reference frames are in `mm`; the position controller works with `meters`
        """

        P = to_coords([x, y], frame)
        target_coords = np.array(
            P.express_coordinates(self.stage_frame), dtype=np.float
        )

        logger.debug(f"target coordinites: {target_coords}")

        with sdc.position.controller() as pos:
            # map m -> mm
            current_coords = np.array((pos.x, pos.y, 0.0)) * 1e3

        delta = target_coords - current_coords

        # convert from mm to m
        delta = delta * 1e-3
        return delta

    def move_stage(
        self,
        x: float,
        y: float,
        frame: Any,
        stage: Any = None,
        threshold: float = 0.0001,
    ):
        """specify target positions in combi reference frame

        Arguments:
            x: wafer x coordinate (`mm`)
            y: wafer y coordinate (`mm`)
            frame: target reference frame (`cell`, `camera`, `laser`)
            stage: stage control interface
            threshold: distance threshold in meters

        Important:
            If a `stage` interface is passed, [move_stage][asdc.client.SDC.move_stage] does not traverse the `z` axis at all!
        """

        def _execute_update(stage, delta, confirm, verbose):

            if confirm:
                input("press enter to allow lateral cell motion...")

            # move horizontally
            stage.update(delta=delta)

            logger.debug(f"stage position: {stage.current_position()}")

        # map position update to position controller frame
        delta = self.compute_position_update(x, y, frame)

        if np.abs(delta).sum() > threshold:
            logger.debug(f"position update: {delta} (mm)")

            # if self.notify:
            #     slack.post_message(f'*confirm update*: (delta={delta})')

            if stage is None:
                with sdc.position.sync_z_step(
                    height=self.step_height, speed=self.speed
                ) as stage:
                    _execute_update(stage, delta, self.confirm, self.verbose)
            else:
                _execute_update(stage, delta, self.confirm, self.verbose)

        if self.initialize_z_position:
            # TODO: define the lower z baseline after the first move

            input("*initialize z position*: press enter to continue...")
            self.initialize_z_position = False

        # update internal tracking of stage position
        if stage is None:
            with sdc.position.controller() as stage:
                self.v_position = stage.current_position()
        else:
            self.v_position = stage.current_position()

        return

    def move(self, x: float, y: float, reference_frame="cell"):
        """slack bot command to move the stage

        A thin json wrapper for [move_stage][asdc.client.SDC.move_stage].

        Arguments:

            - `x`: wafer x coordinate (`mm`)
            - `y`: wafer y coordinate (`mm`)
            - `reference_frame`: target reference frame (`cell`, `camera`, `laser`)
        """

        if self.verbose:
            logger.debug(f"local vars (move): {locals()}")

        frame = {
            "cell": self.cell_frame,
            "laser": self.laser_frame,
            "camera": self.camera_frame,
            "xray": self.xray_frame,
        }[reference_frame]

        self.move_stage(x, y, frame)

        # @ctl -- update the semaphore in the controller process
        # await self.dm_controller(web_client, '<@UHNHM7198> update position is set.')

    def _scale_flow(self, rates: Dict, nominal_rate: float = 0.5) -> Dict:
        """ high nominal flow_rate for running out to steady state """

        total_rate = sum(rates.values())

        if total_rate <= 0.0:
            total_rate = 1.0

        return {key: val * nominal_rate / total_rate for key, val in rates.items()}

    def debug_reglo_droplet(
        self,
        prep_height=0.004,
        wetting_height=0.0011,
        fill_rate=1.0,
        fill_counter_ratio=0.75,
        fill_time=None,
        shrink_counter_ratio=1.1,
        shrink_time=None,
        flow_rate=0.5,
        target_rate=0.05,
        cleanup_duration=3,
        cleanup_pulse_duration=0,
        stage_speed=0.001,
    ):
        """slack bot command for prototyping droplet contact routine

        #### json arguments

        | Name             | Type  | Description                                         | Default |
        |------------------|-------|-----------------------------------------------------|---------|
        | `prep_height`    | float | z setting to grow the droplet                       |     4mm |
        | `wetting_height` | float | z setting to wet the droplet to the surface         |   1.1mm |
        | `fill_rate`      | float | pumping rate during droplet growth                  | 1 mL/min |
        | `fill_counter_ratio` | float | counterpumping ratio during droplet growth          |    0.75 |
        | `fill_time`      | float | droplet growth duration (s)                         |    None |
        | `shrink_counter_ratio` | float | counterpumping ratio during droplet wetting phase   |     1.1 |
        | `shrink_time`    | float | droplet wetting duration (s)                        |    None |
        | `flow_rate`      | float | total flow rate during droplet formation (mL/min)   |     0.5 |
        | `target_rate`    | float | final flow rate after droplet formation  (mL/min)   |    0.05 |
        | `cleanup`        | float | duration of pre-droplet-formation cleanup siphoning |       0 |
        | `stage_speed`    | float | stage velocity during droplet formation op          |   0.001 |

        """

        # stage speed is specified in m/s
        stage_speed = min(stage_speed, 1e-3)
        stage_speed = max(stage_speed, 1e-5)

        # start at zero
        with sdc.position.sync_z_step(height=wetting_height, speed=stage_speed):

            if cleanup_duration > 0:
                # TODO: turn on the needle
                # make an option to pulse loop and dump simultaneously, same rate opposite directions?
                logger.debug("cleaning up...")
                self.reglo.continuousFlow(-10.0, channel=Channel.NEEDLE)
                self.reglo.stop([Channel.SOURCE, Channel.LOOP])

                if cleanup_pulse_duration > 0:
                    pulse_flowrate = -1.0
                    # self.reglo.continuousFlow(pulse_flowrate, channel=Channel.LOOP)
                    self.reglo.continuousFlow(pulse_flowrate, channel=Channel.DRAIN)
                    time.sleep(cleanup_pulse_duration)

                self.reglo.stop(channel=Channel.DRAIN)

                time.sleep(cleanup_duration)

            height_difference = prep_height - wetting_height
            height_difference = max(0, height_difference)
            with sdc.position.sync_z_step(height=height_difference, speed=stage_speed):

                # counterpump slower to fill the droplet
                logger.debug("filling droplet")
                self.reglo.set_rates(
                    {
                        Channel.SOURCE: fill_rate,
                        Channel.LOOP: -fill_rate,
                        Channel.DRAIN: -fill_counter_ratio * fill_rate,
                    }
                )

                fill_start = time.time()
                if fill_time is None:
                    input("*filling droplet*: press enter to continue...")
                else:
                    time.sleep(fill_time)
                fill_time = time.time() - fill_start

            # drop down to wetting height
            # counterpump faster to shrink the droplet
            logger.debug("shrinking droplet")
            shrink_flowrate = fill_rate * shrink_counter_ratio
            self.reglo.continuousFlow(-shrink_flowrate, channel=Channel.DRAIN)

            shrink_start = time.time()
            if shrink_time is None:
                input("*shrinking droplet*: press enter to continue...")
            else:
                time.sleep(shrink_time)
            shrink_time = time.time() - shrink_start

            logger.debug("equalizing differential pumping rate")
            self.reglo.continuousFlow(-fill_rate, channel=Channel.DRAIN)

        # drop down to contact height
        # instructions['fill_time'] = fill_time
        # instructions['shrink_time'] = shrink_time

        time.sleep(3)

        # purge...
        logger.debug("purging solution")
        purge_rate = 11.0
        self.reglo.set_rates(
            {
                Channel.SOURCE: purge_rate,
                Channel.LOOP: -purge_rate,
                Channel.DRAIN: -purge_rate,
            }
        )

        time.sleep(30)

        # reverse the loop direction
        self.reglo.continuousFlow(6.0, channel=Channel.LOOP)

        time.sleep(3)

        # disable source and dump
        self.reglo.stop([Channel.SOURCE, Channel.DRAIN])

        # step to target flow rate
        self.reglo.set_rates({Channel.LOOP: target_rate, Channel.NEEDLE: -2.0})

        # message = f"contact routine with {json.dumps(locals())}"
        # logger.debug(message)
        logger.debug(f"local vars: {locals}")
        return

    def establish_droplet(
        self,
        flow_instructions: Dict = {},
        x_wafer: Optional[float] = None,
        y_wafer: Optional[float] = None,
        logfile: str = None,
    ):
        """Form a new droplet with composition specified by `flow_instructions`.

        if both `x_wafer` and `y_wafer` are specified, the cell will move to these sample coordinates before forming a droplet

        Arguments:
            flow_instructions: specification of droplet composition and loop flow rates.
                Example: `{"op": "set_flow", "pH": 10, "flow_rate": 1.25, "relative_rates": {"NaCl": 1.0}, "purge_time": 90}`
            x_wafer: sample x coordinate to move to before forming a droplet
            y_wafer: sample y coordinate to move to before forming a droplet

        """

        def check_syringe_levels(
            volume_needed: Dict[str, float],
            levels: Dict[str, float],
            headroom: float = 5,
        ) -> List[str]:
            """ compare the volume needed for a push with the current pump levels """
            surplus = {
                key: levels[key] - volume_needed[key] for key in volume_needed.keys()
            }

            # trigger a refill if there won't be more than 5 mL (default)  headroom after the push
            to_refill = [key for key, value in surplus.items() if value < headroom]
            return to_refill

        relative_rates = flow_instructions.get("relative_rates")
        target_rate = float(flow_instructions.get("flow_rate", 1.0))
        purge_time = float(flow_instructions.get("purge_time", 30))
        pH_target = float(flow_instructions.get("pH"))

        # some hardcoded configuration
        pulse_flowrate = -10.0
        purge_rate = 11.0
        purge_ratio = 0.95
        purge_rates = self._scale_flow(relative_rates, nominal_rate=purge_rate)

        # compute required volumes in mL
        volume_needed = {
            key: purge_time * rate / 60 for key, rate in purge_rates.items()
        }
        logger.info(f"solution push target: {volume_needed}")

        levels = self.pump_array.levels()
        logger.info(f"current solution levels: {levels}")

        to_refill = check_syringe_levels(volume_needed, levels)

        while len(to_refill) > 0:
            pump_ids = [
                f"{name} (pump {self.pump_array.get_pump_id(name)})"
                for name in to_refill
            ]
            pump_ids = ", ".join(pump_ids)
            logger.warning(f"Refill and reset syringes: {pump_ids}")
            input("refill and reset pumps to proceed")
            levels = self.pump_array.levels()
            to_refill = check_syringe_levels(volume_needed, levels)

        # droplet workflow -- start at zero
        logger.debug("starting droplet workflow")

        self.reglo.set_rates({Channel.RINSE: 5.0, Channel.NEEDLE: -10.0})
        time.sleep(1)

        with sdc.position.sync_z_step(height=self.wetting_height, speed=self.speed):

            if self.cleanup_pause > 0:
                logger.debug("cleaning up...")
                self.reglo.set_rates({Channel.DRAIN: -10.0})
                self.reglo.stop(Channel.LOOP)

                if self.cleanup_pulse_duration > 0:
                    self.reglo.continuousFlow(pulse_flowrate, channel=Channel.DRAIN)
                    time.sleep(self.cleanup_pulse_duration)

                    self.reglo.stop(channel=Channel.DRAIN)

                time.sleep(self.cleanup_pause / 2)
                self.reglo.stop(Channel.RINSE)
                time.sleep(self.cleanup_pause / 2)
                self.reglo.stop(Channel.DRAIN)

            height_difference = self.droplet_height - self.wetting_height
            height_difference = max(0, height_difference)
            with sdc.position.sync_z_step(
                height=height_difference, speed=self.speed
            ) as stage:

                if x_wafer is not None and y_wafer is not None:
                    self.move_stage(x_wafer, y_wafer, self.cell_frame)

                logger.debug("starting rinse")
                self.reglo.set_rates({Channel.RINSE: 5.0})
                time.sleep(2)

                # counterpump slower to fill the droplet
                logger.debug("filling droplet")
                cell_fill_rates = self._scale_flow(
                    relative_rates, nominal_rate=self.fill_rate
                )
                self.pump_array.set_rates(cell_fill_rates, start=True, fast=True)
                self.reglo.set_rates(
                    {
                        # Channel.SOURCE: self.fill_rate,
                        Channel.LOOP: -self.fill_rate,
                        Channel.DRAIN: -self.fill_counter_ratio * self.fill_rate,
                    }
                )
                time.sleep(self.fill_time / 2)
                self.reglo.stop(Channel.RINSE)
                time.sleep(self.fill_time / 2)

            # drop down to wetting height
            # counterpump faster to shrink the droplet
            logger.debug("differentially pumping to shrink the droplet")
            shrink_flowrate = self.fill_rate * self.shrink_counter_ratio
            self.reglo.continuousFlow(-shrink_flowrate, channel=Channel.DRAIN)
            time.sleep(self.shrink_time)

            logger.debug("equalizing differential pumping rate")
            self.reglo.continuousFlow(-self.fill_rate, channel=Channel.DRAIN)

        # drop down to contact...
        time.sleep(3)

        # purge... (and monitor pH)
        if logfile is None:
            logfile = os.path.join(self.data_dir, "purge.csv")

        with self.phmeter.monitor(interval=1, logfile=logfile):
            logger.debug("purging solution")
            self.pump_array.set_rates(purge_rates, start=True, fast=True)
            self.reglo.set_rates(
                {
                    Channel.LOOP: -purge_ratio * purge_rate,
                    Channel.DRAIN: -purge_ratio * purge_rate,
                }
            )

            time.sleep(purge_time)

            # reverse the loop direction
            self.reglo.continuousFlow(6.0, channel=Channel.LOOP)

            time.sleep(3)

            logger.debug(f"stepping flow rates to {target_rate}")
            self.reglo.set_rates({Channel.LOOP: target_rate, Channel.NEEDLE: -2.0})
            self.pump_array.stop_all(fast=True)
            self.reglo.stop(Channel.DRAIN)

        current_pH_reading = self.phmeter.pH[-1]
        if pH_target is not None:
            pH_error = abs(current_pH_reading - pH_target)
        else:
            pH_error = 0

        if pH_error > 0.5:
            logger.warning(
                f"current pH reading of {current_pH_reading} does not match target of {pH_target}"
            )
        else:
            logger.info(
                f"current pH reading is {current_pH_reading} (target is {pH_target})"
            )

        return

    def syringe_establish_droplet(
        self, x_wafer: float, y_wafer: float, flow_instructions: Dict
    ):
        """ align the stage with a sample point, form a droplet, and flush lines if needed """

        rates = flow_instructions.get("rates")
        cell_fill_rates = self._scale_flow(rates, nominal_rate=0.5)
        line_flush_rates = self._scale_flow(rates, nominal_rate=1.0)

        # if relative flow rates don't match, purge solution
        line_flush_duration = flow_instructions.get("hold_time", 0)
        line_flush_needed = relative_flow(rates) != relative_flow(
            self.pump_array.flow_setpoint
        )

        # droplet workflow -- start at zero
        logger.debug("starting droplet workflow")
        with sdc.position.sync_z_step(
            height=self.wetting_height, speed=self.speed
        ) as stage:

            if self.cleanup_pause > 0:
                logger.debug("cleaning up...")
                self.pump_array.stop_all(counterbalance="full", fast=True)
                time.sleep(self.cleanup_pause)

            self.move_stage(x_wafer, y_wafer, self.cell_frame)

            height_difference = self.droplet_height - self.wetting_height
            height_difference = max(0, height_difference)
            with sdc.position.sync_z_step(height=height_difference, speed=self.speed):

                # counterpump slower to fill the droplet
                logger.debug("differentially pumping to grow the droplet")
                self.pump_array.set_rates(
                    cell_fill_rates,
                    counterpump_ratio=self.fill_ratio,
                    start=True,
                    fast=True,
                )
                time.sleep(self.fill_time)

            # drop down to wetting height
            # counterpump faster to shrink the droplet
            logger.debug("differentially pumping to shrink the droplet")
            self.pump_array.set_rates(
                cell_fill_rates,
                counterpump_ratio=self.shrink_ratio,
                start=True,
                fast=True,
            )
            time.sleep(self.shrink_time)

            logger.debug("equalizing differential pumping rate")
            self.pump_array.set_rates(
                line_flush_rates, counterpump_ratio=0.95, start=True, fast=True
            )

        # flush lines with cell in contact
        if line_flush_needed:
            logger.debug("performing line flush")
            time.sleep(line_flush_duration)

        time.sleep(3)

        logger.debug(f"stepping flow rates to {rates}")
        self.pump_array.set_rates(rates, counterpump_ratio=0.95, start=True, fast=True)

        return

    def quick_expt(
        self,
        instructions: Union[Dict, List[Dict]],
        internal=False,
        plot=True,
        remeasure_ocp=False,
    ):
        """ run a one-off e-chem sequence without touching the stages or pumps """

        logger.info(f"running experiment {instructions}")

        if type(instructions) is dict:
            instructions = [instructions]

        if internal:
            cell = "INTERNAL"
        else:
            cell = "EXTERNAL"

        meta = {
            "instructions": json.dumps(instructions),
            "cell": cell,
            "x_combi": None,
            "y_combi": None,
            "x_versa": self.v_position[0],
            "y_versa": self.v_position[1],
            "z_versa": self.v_position[2],
        }

        with self.db as tx:
            location_id = tx["location"].insert(meta)
            summary = "-".join(step["op"] for step in instructions)
            message = f"location *{location_id}*:  {summary}"
            self.send_notification(message, block=self.confirm_experiment)

        # run e-chem experiments and store results in external csv file
        basename = f"asdc_data_{location_id:03d}"

        # results, metadata = sdc.experiment.run(instructions, cell=cell, verbose=self.verbose, remeasure_ocp=remeasure_ocp)

        with sdc.potentiostat.controller(start_idx=potentiostat_id) as pstat:
            for sequence_id, instruction in enumerate(instructions):

                experiment = sdc.experiment.from_command(instruction)

                opname = instruction["op"]
                metadata = {
                    "op": opname,
                    "location_id": location_id,
                    "datafile": f"{basename}_{sequence_id}_{opname}.csv",
                }

                results, m = pstat.run(experiment)
                status = results.check_quality()
                metadata.update(m)

                with self.db as tx:
                    experiment_id = tx["experiment"].insert(metadata)
                    results.to_csv(os.path.join(self.data_dir, metadata["datafile"]))

                if plot:
                    figpath = os.path.join(
                        self.figure_dir, f"{opname}_plot_{location_id}.png"
                    )
                    save_plot(
                        results,
                        figpath,
                        post_slack=True,
                        title=f"{opname} {location_id}",
                    )

        logger.info("finished experiment")

    def send_notification(self, message, block=False):

        if block:
            message = f"*confirm*: {message}"

        logger.info(message)

        if block:
            input("press enter to allow running the experiment...")

    def run_experiment(self, instructions: List[Dict], plot=True):
        """run an SDC experiment

        args should contain a sequence of SDC experiments -- basically the "instructions"
        segment of an autoprotocol protocol
        that comply with the SDC experiment schema (TODO: finalize and enforce schema)

        TODO: define heuristic checks (and hard validation) as part of the experimental protocol API
        # heuristic check for experimental error signals?
        """

        # check for an instruction group name/intent
        intent = instructions[0].get("intent")

        if intent is not None:
            header = instructions[0]
            instructions = instructions[1:]

        x_combi, y_combi = header.get("x"), header.get("y")

        flow_instructions = instructions[0]
        self.establish_droplet(flow_instructions, x_combi, y_combi)

        meta = {
            "intent": intent,
            "instructions": json.dumps(instructions),
            "x_combi": float(x_combi),
            "y_combi": float(y_combi),
            "x_versa": self.v_position[0],
            "y_versa": self.v_position[1],
            "z_versa": self.v_position[2],
            "flag": False,
        }

        with self.db as tx:
            location_id = tx["location"].insert(meta)
            summary = "-".join(step["op"] for step in instructions)
            message = f"location *{location_id}*:  {summary}"
            self.send_notification(message, block=self.confirm_experiment)

        # run e-chem experiments and store results in external csv file
        basename = f"asdc_data_{location_id:03d}"
        pH_logfile = os.path.join(self.data_dir, f"pH_log_run{location_id:03d}.csv")

        with self.phmeter.monitor(interval=5, logfile=pH_logfile):
            with sdc.potentiostat.controller(start_idx=potentiostat_id) as pstat:
                for sequence_id, instruction in enumerate(instructions):

                    logger.debug(f"running {instruction}")

                    opname = instruction.get("op")
                    if opname is None:
                        continue

                    logger.info(f"running {opname}")

                    experiment = sdc.experiment.from_command(instruction)

                    if experiment is None:
                        continue

                    metadata = {
                        "op": opname,
                        "location_id": location_id,
                        "datafile": f"{basename}_{sequence_id}_{opname}.csv",
                    }

                    results, m = pstat.run(experiment)

                    try:
                        status = results.check_quality()
                    except Exception as err:
                        logger.error(f"data check: {err}")

                    metadata.update(m)

                    if self.pump_array:
                        metadata["flow_setpoint"] = json.dumps(
                            self.pump_array.flow_setpoint
                        )

                    with self.db as tx:
                        experiment_id = tx["experiment"].insert(metadata)
                        results.to_csv(
                            os.path.join(self.data_dir, metadata["datafile"])
                        )

                    if plot:
                        figpath = os.path.join(
                            self.figure_dir,
                            f"{opname}_plot_{location_id}_{sequence_id}.png",
                        )
                        save_plot(
                            results,
                            figpath,
                            post_slack=self.plot_slack,
                            title=f"{opname} {location_id}",
                        )

        logger.info(f"finished experiment {location_id}: {summary}")

    def run_characterization(self, args: str):
        """perform cell cleanup and characterization

        the header instruction should contain a list of primary keys
        corresponding to sample points that should be characterized.

        run_characterization [
            {"intent": "characterize", "experiment_id": 22},
            {"op": "surface-cam"}
            {"op": "laser-reflectance"}
            {"op": "xrays"}
        ]

        """

        # the header block should contain the `experiment_id`
        # for the spots to be characterized
        instructions = json.loads(args)

        header = instructions[0]
        instructions = instructions[1:]

        # check for an instruction group name/intent
        intent = header.get("intent")
        experiment_id = header.get("experiment_id")

        # get all relevant samples
        samples = self.db["experiment"].find(
            experiment_id=experiment_id, intent="deposition"
        )

        if instructions[0].get("op") == "set_flow":
            flow_instructions = instructions[0]
            for sample in samples:
                x_combi = sample.get("x_combi")
                y_combi = sample.get("y_combi")
                primary_key = sample.get("id")

                self.establish_droplet(flow_instructions, x_combi, y_combi)

        # run cleanup and optical characterization
        self.pump_array.stop_all(counterbalance="full", fast=True)
        time.sleep(0.25)

        characterization_ops = set(i.get("op") for i in instructions if "op" in i)

        with sdc.position.sync_z_step(height=self.wetting_height, speed=self.speed):

            if self.cleanup_pause > 0:
                time.sleep(self.cleanup_pause)

            height_difference = self.characterization_height - self.wetting_height
            height_difference = max(0, height_difference)
            with sdc.position.sync_z_step(height=height_difference, speed=self.speed):

                # run laser and camera scans
                samples = self.db["experiment"].find(
                    experiment_id=experiment_id, intent="deposition"
                )
                for idx, sample in enumerate(samples):

                    x_combi = sample.get("x_combi")
                    y_combi = sample.get("y_combi")
                    primary_key = sample.get("id")

                    if "surface-cam" in characterization_ops:
                        if self.notify:
                            web_client.chat_postMessage(
                                channel="#asdc",
                                text=f"inspecting deposit quality",
                                icon_emoji=":sciencebear:",
                            )

                        self.move_stage(x_combi, y_combi, self.camera_frame)
                        self.capture_image(primary_key=primary_key)

                        image_name = f"deposit_pic_{primary_key:03d}.png"
                        figpath = os.path.join(self.data_dir, image_name)
                        try:
                            _slack.post_image(
                                web_client, figpath, title=f"deposit {primary_key}"
                            )
                        except:
                            pass

                    if self.notify:
                        web_client.chat_postMessage(
                            channel="#asdc",
                            text=f"acquiring laser reflectance data",
                            icon_emoji=":sciencebear:",
                        )

                    with sdc.position.sync_z_step(
                        height=self.laser_scan_height, speed=self.speed
                    ) as stage:

                        # laser scan
                        if "laser-reflectance" in characterization_ops:

                            self.move_stage(
                                x_combi, y_combi, self.laser_frame, stage=stage
                            )
                            self.reflectance(primary_key=primary_key, stage=stage)

                        # xray scan
                        if "xrays" in characterization_ops:

                            self.move_stage(x_combi, y_combi, self.xray_frame)
                            time.sleep(1)

                            prefix = f"sdc-26-{primary_key:04d}"
                            logger.info(f"starting x-rays for {prefix}")
                            epics.dispatch_xrays(
                                prefix, os.path.join(self.data_dir, "xray")
                            )

                self.move_stage(x_combi, y_combi, self.cell_frame)

        self.pump_array.counterpump.stop()

    def xrays(self, args: str):
        """perform 06BM x-ray routine

        `@sdc xrays {"experiment_id": 1}

        the header instruction should contain a list of primary keys
        corresponding to sample points that should be characterized.
        """

        # the header block should contain
        instructions = json.loads(args)
        experiment_id = instructions.get("experiment_id")

        # get all relevant samples
        samples = self.db["experiment"].find(experiment_id=experiment_id)

        with sdc.position.sync_z_step(height=self.xrays_height, speed=self.speed):
            for sample in samples:
                logger.debug("xrd")
                web_client.chat_postMessage(
                    channel="#asdc",
                    text=f"x-ray ops go here...",
                    icon_emoji=":sciencebear:",
                )
                x_combi = sample.get("x_combi")
                y_combi = sample.get("y_combi")
                primary_key = sample.get("id")
                self.move_stage(x_combi, y_combi, self.xray_frame)
                time.sleep(1)

                prefix = f"sdc-26-{primary_key:04d}"
                logger.debug(f"starting x-rays for {prefix}")
                epics.dispatch_xrays(prefix, os.path.join(self.data_dir, "xray"))

            # move back to the cell frame for the second spot
            self.move_stage(x_combi, y_combi, self.cell_frame)

    def droplet(self, args: str):
        """slack bot command for prototyping droplet contact routine

        #### json arguments

        | Name             | Type  | Description                                         | Default |
        |------------------|-------|-----------------------------------------------------|---------|
        | `prep_height`    | float | z setting to grow the droplet                       |     4mm |
        | `wetting_height` | float | z setting to wet the droplet to the surface         |   1.1mm |
        | `fill_rate`      | float | counterpumping ratio during droplet growth          |    0.75 |
        | `fill_time`      | float | droplet growth duration (s)                         |    None |
        | `shrink_rate`    | float | counterpumping ratio during droplet wetting phase   |     1.1 |
        | `shrink_time`    | float | droplet wetting duration (s)                        |    None |
        | `flow_rate`      | float | total flow rate during droplet formation (mL/min)   |     0.5 |
        | `target_rate`    | float | final flow rate after droplet formation  (mL/min)   |    0.05 |
        | `cleanup`        | float | duration of pre-droplet-formation cleanup siphoning |       0 |
        | `stage_speed`    | float | stage velocity during droplet formation op          |   0.001 |
        | `solutions`      | List[str]   | list of solutions to pump with                |   None  |

        """
        instructions = json.loads(args)

        prep_height = max(0, instructions.get("height", 0.004))
        wetting_height = max(0, instructions.get("wetting_height", 0.0011))
        fill_ratio = instructions.get("fill_rate", 0.75)
        fill_time = instructions.get("fill_time", None)
        shrink_ratio = instructions.get("shrink_rate", 1.1)
        shrink_time = instructions.get("shrink_time", None)
        flow_rate = instructions.get("flow_rate", 0.5)
        target_rate = instructions.get("target_rate", 0.05)
        cleanup_duration = instructions.get("cleanup", 0)
        stage_speed = instructions.get("stage_speed", self.speed)
        solutions = instructions.get("solutions")

        # stage speed is specified in m/s
        stage_speed = min(stage_speed, 1e-3)
        stage_speed = max(stage_speed, 1e-5)

        # just pump from the first syringe pump
        # solution = next(iter(self.solutions))
        if solutions is None:
            solution = self.solutions[0]
            s = next(iter(solution))
            _rates = {s: flow_rate}
        elif type(solutions) is list:
            _rates = {s: 1.0 for s in solutions}
        elif type(solutions) is dict:
            _rates = solutions

        rates = self._scale_flow(_rates, nominal_rate=flow_rate)
        target_rates = self._scale_flow(_rates, nominal_rate=target_rate)

        logger.info(f"rates: {rates}")
        logger.info(f"target_rates: {target_rates}")

        # start at zero
        with sdc.position.z_step(height=wetting_height, speed=stage_speed):

            if cleanup_duration > 0:
                logger.info("cleaning up...")
                self.pump_array.stop_all(counterbalance="full", fast=True)
                time.sleep(cleanup_duration)

            height_difference = prep_height - wetting_height
            height_difference = max(0, height_difference)
            with sdc.position.z_step(height=height_difference, speed=stage_speed):

                # counterpump slower to fill the droplet
                logger.info("filling droplet")
                self.pump_array.set_rates(
                    rates, counterpump_ratio=fill_ratio, start=True, fast=True
                )
                fill_start = time.time()
                if fill_time is None:
                    input("*filling droplet*: press enter to continue...")
                else:
                    time.sleep(fill_time)
                fill_time = time.time() - fill_start

            # drop down to wetting height
            # counterpump faster to shrink the droplet
            logger.info("shrinking droplet")
            self.pump_array.set_rates(rates, counterpump_ratio=shrink_ratio, fast=True)
            shrink_start = time.time()
            if shrink_time is None:
                input("*shrinking droplet*: press enter to continue...")
            else:
                time.sleep(shrink_time)
            shrink_time = time.time() - shrink_start

            logger.info("equalizing differential pumping rate")
            self.pump_array.set_rates(rates, fast=True, start=True)

        # drop down to contact height
        instructions["fill_time"] = fill_time
        instructions["shrink_time"] = shrink_time

        time.sleep(3)

        logger.info(f"stepping flow rates to {rates}")
        self.pump_array.set_rates(
            target_rates, counterpump_ratio=0.95, fast=True, start=True
        )

        message = f"contact routine with {json.dumps(instructions)}"
        web_client.chat_postMessage(
            channel="#asdc", text=message, icon_emoji=":sciencebear:"
        )

        return

    def flag(self, primary_key: int):
        """ mark a datapoint as bad """

        with self.db as tx:
            tx["experiment"].update({"id": primary_key, "flag": True}, ["id"])

    def coverage(self, primary_key: int, coverage_estimate: float):
        """ record deposition coverage on (0.0,1.0). """

        if coverage_estimate < 0.0 or coverage_estimate > 1.0:
            _slack.post_message(
                f":terriblywrong: *error:* coverage estimate should be in the range (0.0, 1.0)"
            )
        else:
            with self.db as tx:
                tx["experiment"].update(
                    {"id": primary_key, "coverage": coverage_estimate}, ["id"]
                )

    def refl(self, primary_key: int, reflectance_readout: float):
        """ record the reflectance of the deposit (0.0,inf). """

        if reflectance_readout < 0.0:
            _slack.post_message(
                f":terriblywrong: *error:* reflectance readout should be positive"
            )
        else:
            with self.db as tx:
                tx["experiment"].update(
                    {"id": primary_key, "reflectance": reflectance_readout}, ["id"]
                )

    def reflectance_linescan(
        self, stepsize: float = 0.00015, n_steps: int = 32, stage: Any = None
    ) -> Tuple[List[float], List[float]]:
        """perform a laser reflectance linescan

        Arguments:
            stepsize: distance between linescan measurements (meters)
            n_steps: number of measurements in the scan
            stage: stage controller

        Returns:
            mean: list of reflectance values forming the linescan
            var:  uncertainty for reflectances in the linescan

        Warning:
            `reflectance_linescan` translates the sample stage.
            Ensure that the z-stage is such that the cell is not in contact
            with the sample to avoid dragging, which could potentially damage
            the sample or the cell.
        """
        mean, var = [], []
        if stage is None:
            with sdc.position.controller(speed=self.speed) as stage:

                for step in range(n_steps):

                    reflectance_data = self.reflectometer.collect(timeout=2)
                    mean.append(reflectance_data)
                    # mean.append(np.mean(reflectance_data))
                    # var.append(np.var(reflectance_data))

                    stage.update_y(-stepsize)
                    time.sleep(0.25)
        else:
            for step in range(n_steps):

                reflectance_data = self.reflectometer.collect(timeout=2)
                mean.append(reflectance_data)
                stage.update_y(-stepsize)
                time.sleep(0.25)

        return mean, var

    def reflectance(self, primary_key=None, stage=None):

        # get the stage position at the start of the linescan
        with sdc.position.controller() as s:
            metadata = {"reflectance_xv": s.x, "reflectance_yv": s.y}

        mean, var = self.reflectance_linescan(stage=stage)

        if primary_key is not None:
            filename = f"deposit_reflectance_{primary_key:03d}.json"

            metadata["id"] = primary_key
            metadata["reflectance_file"] = filename

            with self.db as tx:
                tx["experiment"].update(metadata, ["id"])

            with open(os.path.join(self.data_dir, filename), "w") as f:
                data = {"reflectance": mean, "variance": var}
                json.dump(data, f)

        logger.info(f"reflectance: {mean}")

        return mean

    @contextmanager
    def light_on(self):
        """ context manager to toggle the light on and off for image acquisition """
        self.light.set("on")
        yield
        self.light.set("off")

    def capture_image(self, primary_key=None):
        """capture an image from the webcam.

        pass an experiment index to serialize metadata to db
        """

        with self.light_on():
            camera = cv2.VideoCapture(self.camera_index)
            # give the camera enough time to come online before reading data...
            time.sleep(0.5)
            status, frame = camera.read()

        # BGR --> RGB format
        frame = frame[..., ::-1].copy()

        if primary_key is not None:

            image_name = f"deposit_pic_{primary_key:03d}.png"

            with sdc.position.controller() as stage:
                metadata = {
                    "id": primary_key,
                    "image_xv": stage.x,
                    "image_yv": stage.y,
                    "image_name": image_name,
                }

            with self.db as tx:
                tx["experiment"].update(metadata, ["id"])

        else:
            image_name = "test-image.png"

        imageio.imsave(os.path.join(self.data_dir, image_name), frame)
        camera.release()

        return

    def bubble(self, primary_key: int):
        """ record a bubble in the deposit """

        with self.db as tx:
            tx["experiment"].update({"id": primary_key, "has_bubble": True}, ["id"])

    def comment(self, primary_key: int, text: str):
        """ add a comment """

        row = self.experiment_table.find_one(id=primary_key)

        if row["comment"]:
            comment = row["comment"]
            comment += "; "
            comment += text
        else:
            comment = text

        with self.db as tx:
            tx["experiment"].update({"id": primary_key, "comment": comment}, ["id"])

    def stop_pumps(self):
        """ shut off the syringe and counterbalance pumps """
        self.pump_array.stop_all(counterbalance="off")

    def load_experiments(self, instructions_file=None):
        root_dir = os.path.dirname(self.data_dir)
        if instructions_file is None:
            instructions_file = os.path.join(root_dir, "instructions.json")

        with open(instructions_file, "r") as f:
            instructions = json.load(f)

        if self.resume:
            location_idx = self.db["location"].count()
            logger.info(f"resuming starting at sample location {location_idx}")
            instructions = instructions[location_idx:]

        return instructions

    def batch_execute_experiments(self, instructions_file=None):

        instructions = self.load_experiments(instructions_file)

        for instruction_chain in instructions:
            logger.debug(json.dumps(instruction_chain))
            self.run_experiment(instruction_chain)

        return

    def purge_cell(self):

        return

    def droplet_video(self):

        flowrates = {"flow_rate": 1.0, "relative_rates": {"H2O": 1.0}, "purge_time": 15}

        points = [[0, 15], [15, 15], [15, 0], [0, 0]]

        for x, y in points:
            logger.info(f"visiting {x}, {y}")
            isdc.establish_droplet(flowrates, x, y)
            time.sleep(10)


def sdc_client(config_file: str, resume: bool, zmq_pub: bool, verbose: bool):
    """ set up scanning droplet cell client loading from CONFIG_FILE """

    experiment_root, _ = os.path.split(config_file)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    if config.get("notify_slack", False):
        sh = _slack.SlackHandler(client=web_client)
        sh.setLevel(
            logging.CRITICAL
        )  # only log CRITICAL events to slack until setup is finished
        logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(experiment_root, "isdc.log"))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # specify target file relative to config file
    target_file = config.get("target_file")
    config["target_file"] = os.path.join(experiment_root, target_file)

    data_dir = config.get("data_dir")
    if data_dir is None:
        config["data_dir"] = os.path.join(experiment_root, "data")

    figure_dir = config.get("figure_dir")
    if figure_dir is None:
        config["figure_dir"] = os.path.join(experiment_root, "figures")

    os.makedirs(config["data_dir"], exist_ok=True)
    os.makedirs(config["figure_dir"], exist_ok=True)

    # make sure step_height is positive!
    if config["step_height"] is not None:
        config["step_height"] = abs(config["step_height"])

    logfile = config.get("command_logfile", "commands.log")
    logfile = os.path.join(config["data_dir"], logfile)

    logger.info("connecting to the SDC...")
    sdc_interface = SDC(
        verbose=verbose,
        config=config,
        logfile=logfile,
        token=BOT_TOKEN,
        resume=resume,
        zmq_pub=zmq_pub,
    )

    if config.get("notify_slack", False):
        sh.setLevel(logging.INFO)

    logger.info("connected!")
    return sdc_interface


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SDC client")
    parser.add_argument("configfile", type=str, help="config file")
    parser.add_argument(
        "--no-resume", action="store_true", help="ignore starting from checkpoint"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="set up ZMQ publisher for dashboard"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="include extra debugging output"
    )
    args = parser.parse_args()
    logger.debug(f"{args}")

    resume = not args.no_resume
    logger.debug(f"resume?: {resume}")

    if args.dashboard:
        dashboard_log = open(
            os.path.join(os.path.split(args.configfile)[0], "dashboard.log"), "wb"
        )
        dashboard_proc = subprocess.Popen(
            ["panel", "serve", "asdc/dashboard.py"],
            stdout=dashboard_log,
            stderr=dashboard_log,
        )

    isdc = sdc_client(args.configfile, resume, args.dashboard, args.verbose)
