""" asdc.position: pythonnet .NET interface to VersaSTAT motion controller """

import os
import sys
import time
import numpy as np
from contextlib import contextmanager

ax = [0, 0, 0]


@contextmanager
def controller(ip="192.168.10.11", speed=1e-4):
    """ context manager that wraps position controller class Position. """
    pos = Position(ip=ip, speed=speed)
    try:
        yield pos
    except Exception as exc:
        print("unwinding position controller due to exception.")
        raise exc
    finally:
        pass


class Controller:
    def __init__(self):
        self.Parameters = [1, 2, 3]


class Position:
    """ Interface to the VersaSTAT motion controller library """

    def __init__(self, ip="192.168.10.11", speed=0.0001):
        """ instantiate a Position controller context manager """
        self._ip = ip
        self._speed = speed

        # Set up and connect to the position controller
        self.controller = Controller()
        self.settings = None

        # use this global variable to keep track of state
        # across different instantiations of the position controller shim
        global ax
        self.axis = ax

        self._x = 0
        self._y = 0
        self._z = 0

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        self._speed = speed

    def home(block_interval=1):
        """execute the homing operation, blocking for `block_interval` seconds.

        Warning: this will cause the motion stage to return to it's origin.
        This happens to be the maximum height for the stage...
        """
        time.sleep(1)

    def print_status(self):
        """ print motion controller status for each axis. """
        print("ok")

    def current_position(self):
        """return the current coordinates as a list

        axis.Values holds (position, speed, error)
        """
        return self.axis

    def at_setpoint(self, verbose=False):
        """ check that each axis of the position controller is at its setpoint """
        return True

    def update_single_axis(self, axis=0, delta=0.001, verbose=False, poll_interval=0.1):
        """update position setpoint and busy-wait until the motion controller has finished.

        poll_interval: busy-waiting polling interval (seconds)
        """
        self.axis[axis] += delta

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

        for idx, d in enumerate(delta):
            self.axis[idx] += d

        time.sleep(1)
