""" asdc.control: pythonnet .NET interface to VersaSTAT/VersaSCAN libraries """

import os
import sys
import json
import time
import asyncio
import inspect
import logging
import streamz
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import contextmanager

from streamz.dataframe import DataFrame

import zmq
import zmq.asyncio

# set up and bind zmq publisher socket
DASHBOARD_PORT = 2345
DASHBOARD_ADDRESS = "127.0.0.1"
DASHBOARD_URI = f"tcp://{DASHBOARD_ADDRESS}:{DASHBOARD_PORT}"

context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.PUB)
# socket.bind(DASHBOARD_URI)

shim_data = os.path.join(os.path.split(__file__)[0], "data")
# df = pd.read_csv(os.path.join(shim_data, 'test_data.csv'), index_col=0)
df = pd.read_csv(os.path.join(shim_data, "test_open_circuit.csv"), index_col=0)


class VersaStatError(Exception):
    pass


@contextmanager
def controller(start_idx=17109013, initial_mode="potentiostat"):
    """ context manager that wraps potentiostat controller class Control. """
    ctl = Potentiostat(start_idx=start_idx, initial_mode=initial_mode)
    try:
        ctl.stop()
        ctl.clear()
        yield ctl
    except Exception as exc:
        print(exc)
        print("Exception: unwind potentiostat controller...")
        ctl.stop()
        ctl.clear()
        ctl.disconnect()
        raise
    finally:
        print("disconnect from potentiostat controller.")
        ctl.stop()
        ctl.clear()
        ctl.disconnect()


class Potentiostat:
    """Interface to the VersaSTAT SDK library for instrument control

    methods are broken out into `Immediate` (direct instrument control) and `Experiment`.
    """

    def __init__(self, start_idx=0, initial_mode="potentiostat", poll_interval=1):

        self.instrument = None
        self.start_idx = start_idx
        self.connect()

        self.serial_number = None
        self.model = None
        self.options = None
        self.low_current_interface = None

        self.mode = initial_mode
        self.poll_interval = poll_interval
        self.current_range = None

        # action buffer for shim
        self.action_queue = []

        self._points_available = 0

        # load data from a file...
        self._data = df

    def __call__(self, experiment):
        return self.run(experiment)

    def check_overload(self):
        self.update_status()
        overload_status = self.overload_status()
        if overload_status != 0:
            print("OVERLOAD:", overload_status)
        return overload_status

    def read_buffers(self):
        return {
            "current": self.current(),
            "potential": self.potential(),
            "elapsed_time": self.elapsed_time(),
            "applied_potential": self.applied_potential(),
            "current_range": self.current_range_history(),
            "segment": self.segment(),
        }

    def run(self, experiment):
        """ run an SDC experiment sequence -- busy wait until it's finished """

        # this is a bit magical...
        # `experiment` has an attribute `setup_func` that holds the name of the .NET function
        # that should be invoked to add an experiment
        # `experiment.setup` is responsible for looking it up and invoking it
        # with e.g. `f = getattr(pstat.instrument.Experiment, experiment.setup_func)`
        # this way, an `SDCSequence` can call all the individual `setup` methods

        # no need to run any setup...
        # argstring = experiment.setup(self.instrument.Experiment)
        argstring = str(experiment)

        metadata = {"timestamp_start": datetime.now(), "parameters": argstring}
        self.start()

        error_codes = set()

        # while self.sequence_running():
        n = self._data.shape[0]
        n_iters = 40
        chunk_size = n // 40

        for idx in range(n_iters):
            self.points_available = idx * chunk_size
            time.sleep(self.poll_interval)
            error_codes.add(self.check_overload())
            print(f"points: {self.points_available}")

        metadata["timestamp_end"] = datetime.now()
        metadata["error_codes"] = json.dumps(list(map(int, error_codes)))
        results = self.read_buffers()

        # cast results into specific e-chem result type
        # (which subclass pandas.DataFrame and have a validation and plotting interface)
        results = experiment.marshal(results)

        return results, metadata

    def read_chunk(self, start):
        return pd.DataFrame(
            {
                "current": self.current(start=start),
                "potential": self.potential(start=start),
                "elapsed_time": self.elapsed_time(start=start),
            }
        )

    def stream(self, experiment):
        """ stream the data from the potentiostat... """
        source = streamz.Stream()

        metadata = {"timestamp_start": datetime.now(), "parameters": str(experiment)}
        self.start()

        # build a list of pd.DataFrames
        # to concat into the full measurement data
        chunks = source.sink_to_list()

        # publish the pd.DataFrame chunk over zmq
        send_data = source.sink(lambda x: socket.send_pyobj(x))

        # monitor convergence
        # hacky -- rely on measurement interval being ~1s
        example = pd.DataFrame({"current": [], "potential": [], "elapsed_time": []})
        sdf = DataFrame(source, example=example)
        early_stop = experiment.register_early_stopping(sdf)

        n = self._data.shape[0]
        n_iters = 40
        chunk_size = n // 40

        cursor = 0

        # while self.sequence_running():
        for idx in range(1, n_iters + 1):
            self.points_available += chunk_size
            source.emit(self.read_chunk(cursor))
            cursor += chunk_size

            if experiment.stop_execution:
                print("stopping early")
                break

            time.sleep(self.poll_interval)

        metadata["timestamp_end"] = datetime.now()
        results = self.read_buffers()

        # cast results into specific e-chem result type
        # (which subclass pandas.DataFrame and have a validation and plotting interface)
        results = experiment.marshal(results)

        return results, metadata, chunks

    def connect(self):
        self.index = self.start_idx
        self.connected = True

    def disconnect(self):
        self.connected = False

    # Immediate methods -- direct instrument control

    def set_cell(self, status="on"):
        """ turn the cell on or off """

        if status not in ("on", "off"):
            raise ArgumentError("specify valid cell status in {on, off}")

    def choose_cell(self, choice="external"):
        """ choose between the internal and external cells. """

        if choice not in ("internal", "external"):
            raise ArgumentError("specify valid cell in {internal, external}")

    def set_mode(self, mode):
        """ choose between potentiostat and galvanostat modes. """

        if mode not in ("potentiostat", "galvanostat"):
            raise ArgumentError("set mode = {potentiostat, galvanostat}")

    def set_current_range(self, current_range):

        valid_current_ranges = [
            "2A",
            "200mA",
            "20mA",
            "2mA",
            "200uA",
            "20uA",
            "2uA",
            "200nA",
            "20nA",
            "2nA",
        ]

        if current_range not in valid_current_ranges:
            raise ArgumentError(
                "specify valid current range ({})".format(valid_current_ranges)
            )

        self.current_range = current_range

    def set_dc_potential(self, potential):
        """ Set the output DC potential (in Volts). This voltage must be within the instruments capability."""
        pass

    def set_dc_current(self, current):
        """Set the output DC current (in Amps). This current must be within the instruments capability.

        Calling this method also changes to Galvanostat mode and sets the current range to the correct value.
        WARNING: Once cell is enabled after setting the DC current, do not change to potentiostatic mode or change the current range.
        These will affect the value being applied to the cell.
        """
        pass

    def set_ac_frequency(self, frequency):
        """ Sets the output AC Frequency (in Hz). This frequency must be within the instruments capability."""
        pass

    def set_ac_amplitude(self, amplitude):
        """ Sets the output AC Amplitude (in RMS Volts). This amplitude must be within the instruments capabilities."""
        pass

    def set_ac_waveform(self, mode="on"):
        waveform_modes = ["on", "off"]

        if mode not in waveform_modes:
            raise ArgumentError("specify valid AC waveform mode {on, off}.")

    def update_status(self):
        """Retrieve the status information from the instrument.
        Also auto-ranges the current if an experiment sequence is not in progress.

        Call this prior to calling the status methods below.
        """

        pass

    def latest_potential(self):
        """ get the latest stored E value. """
        return None

    def latest_current(self):
        """ get the latest stored I value. """
        return None

    def overload_status(self, raise_exception=False):
        """check for overloading.
                0 indicates no overload, 1 indicates I (current) Overload, 2
        indicates E, Power Amp or Thermal Overload has occurred.
        """
        overload_cause = {
            1: "I (current) overload",
            2: "E, Power Amp, or Thermal overload",
        }

        # overload_code = self.instrument.Immediate.GetOverload()
        overload_code = 0

        if overload_code and raise_exception:
            msg = "A " + overload_cause[overload_code] + " has occurred."
            raise VersaStatError(msg)

        return overload_code

    def booster_enabled(self):
        """ check status of the booster switch. """
        return None

    def cell_enabled(self):
        """ check status of the cell. """
        return None

    def autorange_current(self, auto):
        """Enable or disable (default is enabled) automatic current ranging while an experiment is not running.
        Disabling auto-ranging is useful when wanting to apply a DC current in immediate mode.
        """
        pass

    # Experiment methods
    # Experiment actions apparently can be run asynchronously

    def actions(self):
        """ get the current experiment action queue. """
        # Returns a list of comma delimited action names that are supported by the instrument that is currently connected
        return None

    def clear(self):
        """ clear the experiment action queue. """
        self.action_queue = []

    def start(self, max_wait_time=30, poll_interval=2):
        """Starts the sequence of actions in the instrument that is currently connected.
        Wait until the instrument starts the action to return control flow."""
        print("started experiment sequence successfully.")

        return

    def stop(self):
        """ Stops the sequence of actions that is currently running in the instrument that is currently connected. """
        pass

    def skip(self):
        """Skips the currently running action and immediately starts the next action.
        If there is no more actions to run, the sequence is simply stopped.
        """
        self.action_queue.pop(0)

    def sequence_running(self):
        """ Returns true if a sequence is currently running on the connected instrument, false if not. """
        pass

    @property
    def points_available(self):
        """Returns the number of points that have been stored by the instrument after a sequence of actions has begun.
        Returns -1 when all data has been retrieved from the instrument.
        """
        return self._points_available

    @points_available.setter
    def points_available(self, value):
        self._points_available = value

    def last_open_circuit(self):
        """Returns the last measured Open Circuit value.
        This value is stored at the beginning of the sequence (and updated anytime the “AddMeasureOpenCircuit” action is called)"""
        return None

    # The following Action Methods can be called in order to create a sequence of Actions.
    # A single string argument encodes multiple parameters as comma-separated lists...
    # For example, AddOpenCircuit( string ) could be called, then AddEISPotentiostatic( string ) called.
    # This would create a sequence of two actions, when started, the open circuit experiment would run, then the impedance experiment.

    # TODO: write a class interface for different experimental actions to streamline logging and serialization?

    # TODO: code-generation for GetData* interface?

    def potential(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available

        return self._data["potential"].values[start:num_points]

    def current(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available

        return self._data["current"].values[start:num_points]

    def elapsed_time(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available

        return self._data["elapsed_time"].values[start:num_points]

    def applied_potential(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available

        return self._data["applied_potential"].values[start:num_points]

    def segment(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available

        return self._data["segment"].values[start:num_points]

    def current_range_history(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available

        return self._data["current_range"].values[start:num_points]

    def hardcoded_open_circuit(self, params):
        default_params = (
            "1,10,NONE,<,0,NONE,<,0,2MA,AUTO,AUTO,AUTO,INTERNAL,AUTO,AUTO,AUTO"
        )
        print(default_params)
        return status, default_params
