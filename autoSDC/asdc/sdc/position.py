""" asdc.position: pythonnet .NET interface to VersaSTAT motion controller """

CONTROLLER_ADDRESS = "192.168.10.11"

import os
import clr
import sys
import time
import asyncio
import functools
import numpy as np
from contextlib import contextmanager, asynccontextmanager

import traceback

# pythonnet checks PYTHONPATH for assemblies to load...
# so add the VeraScan libraries to sys.path
versascan_path = "C:/Program Files (x86)/Princeton Applied Research/VersaSCAN"
sys.path.append(versascan_path)
sys.path.append(os.path.join(versascan_path, "Devices"))

dlls = ["CommsLibrary", "DeviceInterface", "ScanDevices", "NanomotionXCD"]
for dll in dlls:
    clr.AddReference(dll)

clr.AddReference("System")
clr.AddReference("System.Net")

from System.Net import IPAddress
from SolartronAnalytical.DeviceInterface.NanomotionXCD import XCD, XcdSettings


@contextmanager
def controller(ip=CONTROLLER_ADDRESS, speed=1e-4):
    """ context manager that wraps position controller class Position. """
    pos = Position(ip=ip, speed=speed)
    try:
        pos.controller.Connect()
        yield pos
    except Exception as exc:
        print("unwinding position controller due to exception.")
        traceback.print_exc()
        pos.controller.Disconnect()
        raise exc
    finally:
        pos.controller.Disconnect()


@contextmanager
def sync_z_step(ip=CONTROLLER_ADDRESS, height=None, speed=1e-4):
    """wrap position controller context manager

    perform vertical steps before lateral cell motion with the ctx manager
    so that the cell drops back down to baseline z level if the `move` task is completed
    """

    with controller(ip=ip, speed=speed) as pos:

        baseline_z = pos.z

        try:
            if height is not None:
                if height <= 0:
                    raise ValueError("z_step should be positive")
                pos.update_z(delta=height)

            yield pos

        finally:
            if height is not None:
                dz = baseline_z - pos.z
                pos.update_z(delta=dz)


# @contextmanager
# def sync_z_step(height=0.002, ip=CONTROLLER_ADDRESS, speed=1e-4):
#     """ sync controller context manager for z step
#     perform a vertical step with no horizontal movement
#     """

#     if height <= 0:
#         raise ValueError("z_step should be positive")

#     try:

#         with controller(ip=ip, speed=speed) as pos:
#             baseline_z = pos.z
#             pos.update_z(delta=height)

#         yield

#     finally:

#         with controller(ip=ip, speed=speed) as pos:
#             dz = baseline_z - pos.z
#             pos.update_z(delta=dz)


@asynccontextmanager
async def acontroller(loop=None, z_step=None, ip=CONTROLLER_ADDRESS, speed=1e-4):
    """wrap position controller context manager

    perform vertical steps before lateral cell motion with the ctx manager
    so that the cell drops back down to baseline z level if the `move` task is completed
    """

    with controller(ip=ip, speed=speed) as pos:

        baseline_z = pos.z

        try:
            if z_step is not None:

                if z_step <= 0:
                    raise ValueError("z_step should be positive")

                f = functools.partial(pos.update_z, delta=z_step)
                await loop.run_in_executor(None, f)

            yield pos

        finally:

            if z_step is not None:

                dz = baseline_z - pos.z
                f = functools.partial(pos.update_z, delta=dz)
                await loop.run_in_executor(None, f)


@asynccontextmanager
async def z_step(loop=None, height=0.002, ip=CONTROLLER_ADDRESS, speed=1e-4):
    """async controller context manager for z step
    perform a vertical step with no horizontal movement
    """

    if height <= 0:
        raise ValueError("z_step should be positive")

    try:

        with controller(ip=ip, speed=speed) as pos:

            baseline_z = pos.z
            f = functools.partial(pos.update_z, delta=height)
            await loop.run_in_executor(None, f)

        yield

    finally:

        with controller(ip=ip, speed=speed) as pos:

            dz = baseline_z - pos.z
            f = functools.partial(pos.update_z, delta=dz)
            await loop.run_in_executor(None, f)


class Position:
    """ Interface to the VersaSTAT motion controller library """

    def __init__(self, ip="192.168.10.11", speed=0.0001):
        """ instantiate a Position controller context manager """
        self._ip = ip
        self._speed = speed

        # Set up and connect to the position controller
        self.controller = XCD()
        self.settings = XcdSettings()

        self.settings.Speed = self._speed
        self.settings.IPAddress = IPAddress.Parse(self._ip)

        self.controller.Connect()

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        self._speed = speed
        self.settings.Speed = self._speed

    @property
    def x(self):
        """ the current stage x position """
        return self.current_position()[0]

    @property
    def y(self):
        """ the current stage y position """
        return self.current_position()[1]

    @property
    def z(self):
        """ the current stage z position """
        return self.current_position()[2]

    def current_position(self):
        """return the current coordinates as a list

        axis.Values holds (position, speed, error)
        """
        return [axis.Values[0] for axis in self.controller.Parameters]

    def home(block_interval=1):
        """execute the homing operation, blocking for `block_interval` seconds.

        Warning: this will cause the motion stage to return to it's origin.
        This happens to be the maximum height for the stage...
        """
        if not self.controller.IsHomingDone:
            self.controller.DoHoming()
            time.sleep(block_interval)

    def print_status(self):
        """ print motion controller status for each axis. """
        for axis in self.controller.Parameters:
            print(
                "{} setpoint = {} {}".format(axis.Quantity, axis.SetPoint, axis.Units)
            )

            for idx in range(axis.ValueNames.Length):
                print(axis.ValueNames[idx], axis.Values[idx], axis.Units)
                print()

    def at_setpoint(self, verbose=False):
        """ check that each axis of the position controller is at its setpoint """

        for ax in self.controller.Parameters:

            if verbose:
                print(ax.Values[0], ax.Units)

            if not ax.IsAtSetPoint:
                return False

        return True

    def update_single_axis(self, axis=0, delta=0.001, verbose=False, poll_interval=0.1):
        """update position setpoint and busy-wait until the motion controller has finished.

        poll_interval: busy-waiting polling interval (seconds)
        """

        # update the setpoint for the x axis
        for idx, ax in enumerate(self.controller.Parameters):
            if idx == axis:
                if verbose:
                    print(ax.Quantity)

                ax.SetPoint = ax.Values[0] + delta

                break

        # busy-wait while the motion controller moves the stage
        while not self.at_setpoint(verbose=verbose):
            time.sleep(poll_interval)

        return

    def update_x(self, delta=0.001, verbose=False, poll_interval=0.1):
        return self.update_single_axis(
            axis=0, delta=delta, verbose=verbose, poll_interval=poll_interval
        )

    def update_y(self, delta=0.001, verbose=False, poll_interval=0.1):
        return self.update_single_axis(
            axis=1, delta=delta, verbose=verbose, poll_interval=poll_interval
        )

    def update_z(self, delta=0.001, verbose=False, poll_interval=0.1):
        return self.update_single_axis(
            axis=2, delta=delta, verbose=verbose, poll_interval=poll_interval
        )

    def update(
        self,
        delta=[0.001, 0.001, 0.0],
        step_height=None,
        compress=None,
        verbose=False,
        poll_interval=0.1,
        max_wait_time=25,
    ):
        """update position setpoint and busy-wait until the motion controller has finished.

        delta: position update [dx, dy, dz]
        step_height: ease off vertically before updating position
        poll_interval: busy-waiting polling interval (seconds)
        """

        if step_height is not None and step_height > 0:
            step_height = abs(step_height)
            self.update_z(delta=step_height, verbose=verbose)

        for d, ax in zip(delta, self.controller.Parameters):

            if verbose:
                print(ax.Quantity)

            if d != 0.0:
                ax.SetPoint = ax.Values[0] + d

        # busy-wait while the motion controller moves the stage
        time_elapsed = 0
        while not self.at_setpoint(verbose=verbose):

            time.sleep(poll_interval)
            time_elapsed += poll_interval

            if time_elapsed > max_wait_time:
                raise TimeoutError(
                    "Max position update time of {}s exceeded".format(max_wait_time)
                )

        if step_height is not None and step_height > 0:
            self.update_z(delta=-step_height, verbose=verbose)

        if compress is not None and abs(compress) > 0:
            compress = np.clip(abs(compress), 0, 5e-5)

            self.update_z(delta=-compress)
            self.update_z(delta=compress)
