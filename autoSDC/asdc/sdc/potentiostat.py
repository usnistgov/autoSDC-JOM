""" asdc.control: pythonnet .NET interface to VersaSTAT/VersaSCAN libraries """

import os
import clr
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# pythonnet checks PYTHONPATH for assemblies to load...
# so add the VersaSCAN libraries to sys.path
vdkpath = "C:/Program Files (x86)/Princeton Applied Research/VersaSTAT Development Kit"
sys.path.append(vdkpath)

# load instrument control library...
clr.AddReference("VersaSTATControl")
from VersaSTATControl import Instrument


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
        time.sleep(1)
        ctl.clear()
        time.sleep(1)
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

        self.instrument = Instrument()
        self.start_idx = start_idx
        self.connect()

        self.serial_number = self.instrument.GetSerialNumber()
        self.model = self.instrument.GetModel()
        self.options = self.instrument.GetOptions()
        self.low_current_interface = self.instrument.GetIsLowCurrentInterfacePresent()

        self.mode = initial_mode
        self.poll_interval = poll_interval
        self.current_range = None

        return

    def __call__(self, experiment):
        return self.run(experiment)

    def check_overload(self):
        self.update_status()
        overload_status = self.overload_status()
        if overload_status != 0:
            print("OVERLOAD:", overload_status)
        return overload_status

    def read_buffers(self, start: int = 0, eis_mode: bool = False):
        num_points = self.points_available() - start

        results = {
            "current": self.current(start, num_points),
            "potential": self.potential(start, num_points),
            "elapsed_time": self.elapsed_time(start, num_points),
            "applied_potential": self.applied_potential(start, num_points),
            "current_range": self.current_range_history(start, num_points),
            "segment": self.segment(start, num_points),
        }

        if eis_mode:
            results["frequency"] = self.frequency(start, num_points)
            results["impedance_real"] = self.impedance_real(start, num_points)
            results["impedance_imag"] = self.impedance_imag(start, num_points)

        return pd.DataFrame(results)

    def run(self, experiment, clear=True):
        """ run an SDC experiment sequence -- busy wait until it's finished """

        # this is a bit magical...
        # `experiment` has an attribute `setup_func` that holds the name of the .NET function
        # that should be invoked to add an experiment
        # `experiment.setup` is responsible for looking it up and invoking it
        # with e.g. `f = getattr(pstat.instrument.Experiment, experiment.setup_func)`
        # this way, an `SDCSequence` can call all the individual `setup` methods

        argstring = experiment.setup(self.instrument.Experiment)

        # configure data collection with extra tracked variables for EIS
        eis_mode = False
        if "EIS" in experiment.name:
            eis_mode = True

        metadata = {"timestamp_start": datetime.now(), "parameters": argstring}
        self.start()

        error_codes = set()

        source = streamz.Stream()

        # build a list of pd.DataFrames
        # to concat into the full measurement data
        chunks = source.sink_to_list()

        # # publish the pd.DataFrame chunk over zmq
        # send_data = source.sink(lambda x: socket.send_pyobj(x))

        # streaming dataframe for early stopping (and potential error checking) callbacks

        template = {
            "current": [],
            "potential": [],
            "elapsed_time": [],
            "applied_potential": [],
            "current_range": [],
            "segment": [],
        }
        if eis_mode:
            template.update(
                {"frequency": [], "impedance_real": [], "impedance_imag": []}
            )

        example = pd.DataFrame(template)

        sdf = DataFrame(source, example=example)
        early_stop = experiment.register_early_stopping(sdf)

        data_cursor = 0
        stop_flagged = False
        while self.sequence_running():
            time.sleep(self.poll_interval)
            error_codes.add(self.check_overload())
            data_chunk = self.read_buffers(start=data_cursor, eis_mode=eis_mode)
            chunksize, _ = data_chunk.shape
            data_cursor += chunksize

            if chunksize > 0:
                source.emit(data_chunk)

            if experiment.stop_execution and not stop_flagged:
                logger.debug("stopping experiment early")
                self.skip()
                stop_flagged = True

        logger.debug("finished running experiment")
        metadata["timestamp_end"] = datetime.now()
        metadata["error_codes"] = json.dumps(list(map(int, error_codes)))
        # results = self.read_buffers()
        results = pd.concat(chunks, ignore_index=True)

        # cast results into specific e-chem result type
        # (which subclass pandas.DataFrame and have a validation and plotting interface)
        results = experiment.marshal(results)

        self.clear()

        return results, metadata

    def connect(self):
        self.index = self.instrument.FindNext(self.start_idx)
        self.connected = self.instrument.Connect(self.index)
        return

    def disconnect(self):
        self.instrument.Close()

    # Immediate methods -- direct instrument control

    def set_cell(self, status="on"):
        """ turn the cell on or off """

        if status not in ("on", "off"):
            raise ArgumentError("specify valid cell status in {on, off}")

        if status == "on":
            self.instrument.Immediate.SetCellOn()

        else:
            self.instrument.Immediate.SetCellOff()

    def choose_cell(self, choice="external"):
        """ choose between the internal and external cells. """

        if choice not in ("internal", "external"):
            raise ArgumentError("specify valid cell in {internal, external}")

        if choice == "external":
            self.instrument.Immediate.SetCellExternal()

        elif choice == "internal":
            self.instrument.Immediate.SetCellExternal()

    def set_mode(self, mode):
        """ choose between potentiostat and galvanostat modes. """

        if mode not in ("potentiostat", "galvanostat"):
            raise ArgumentError("set mode = {potentiostat, galvanostat}")

        if mode == "potentiostat":
            self.instrument.Immediate.SetModePotentiostat()

        elif mode == "galvanostat":
            self.instrument.Immediate.SetModeGalvanostat()

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

        # dispatch the right SetIRange_* function....
        current = "SetIRange_{}".format(current_range)
        set_current = getattr(self.instrument.Immediate, current)
        set_current()

    def set_dc_potential(self, potential):
        """ Set the output DC potential (in Volts). This voltage must be within the instruments capability."""
        self.instrument.Immediate.SetDCPotential(potential)

    def set_dc_current(self, current):
        """Set the output DC current (in Amps). This current must be within the instruments capability.

        Calling this method also changes to Galvanostat mode and sets the current range to the correct value.
        WARNING: Once cell is enabled after setting the DC current, do not change to potentiostatic mode or change the current range.
        These will affect the value being applied to the cell.
        """
        self.instrument.Immediate.SetDCCurrent(current)

    def set_ac_frequency(self, frequency):
        """ Sets the output AC Frequency (in Hz). This frequency must be within the instruments capability."""
        self.instrument.Immediate.SetACFrequency(frequency)

    def set_ac_amplitude(self, amplitude):
        """ Sets the output AC Amplitude (in RMS Volts). This amplitude must be within the instruments capabilities."""
        self.instrument.Immediate.SetACAmplitude(amplitude)

    def set_ac_waveform(self, mode="on"):
        waveform_modes = ["on", "off"]

        if mode not in waveform_modes:
            raise ArgumentError("specify valid AC waveform mode {on, off}.")

        if mode == "on":
            self.instrument.Immediate.SetACWaveformOn()
        elif mode == "off":
            self.instrument.Immediate.SetACWaveformOff()

    def update_status(self):
        """Retrieve the status information from the instrument.
        Also auto-ranges the current if an experiment sequence is not in progress.

        Call this prior to calling the status methods below.
        """

        self.instrument.Immediate.UpdateStatus()

    def latest_potential(self):
        """ get the latest stored E value. """
        return self.instrument.Immediate.GetE()

    def latest_current(self):
        """ get the latest stored I value. """
        return self.instrument.Immediate.GetI()

    def overload_status(self, raise_exception=False):
        """check for overloading.
                0 indicates no overload, 1 indicates I (current) Overload, 2
        indicates E, Power Amp or Thermal Overload has occurred.
        """
        overload_cause = {
            1: "I (current) overload",
            2: "E, Power Amp, or Thermal overload",
        }

        overload_code = self.instrument.Immediate.GetOverload()

        if overload_code and raise_exception:
            msg = "A " + overload_cause[overload_code] + " has occurred."
            raise VersaStatError(msg)

        return overload_code

    def booster_enabled(self):
        """ check status of the booster switch. """
        return self.instrument.Immediate.GetBoosterEnabled()

    def cell_enabled(self):
        """ check status of the cell. """
        return self.instrument.Immediate.GetCellEnabled()

    def autorange_current(self, auto):
        """Enable or disable (default is enabled) automatic current ranging while an experiment is not running.
        Disabling auto-ranging is useful when wanting to apply a DC current in immediate mode.
        """
        if auto:
            self.instrument.Immediate.SetAutoIRangeOn()
        else:
            self.instrument.Immediate.SetAutoIRangeOff()

    # Experiment methods
    # Experiment actions apparently can be run asynchronously

    def actions(self):
        """ get the current experiment action queue. """
        # Returns a list of comma delimited action names that are supported by the instrument that is currently connected
        action_list = self.instrument.Experiment.GetActionList()
        return action_list.split(",")

    def clear(self):
        """ clear the experiment action queue. """
        self.instrument.Experiment.Clear()

    def start(self, max_wait_time=30, poll_interval=2):
        """Starts the sequence of actions in the instrument that is currently connected.
        Wait until the instrument starts the action to return control flow."""

        self.instrument.Experiment.Start()

        # Note: ctl.start() can return before the sequence actually starts running,
        # so it's possible to skip right past the data collection spin-waiting loop
        # which writes a data-less log file and pushes the next experiment onto the queue
        # while the instrument is still going on with the current one.
        # it appears that this is not safe....
        elapsed = 0

        while not self.sequence_running():
            time.sleep(poll_interval)
            elapsed += poll_interval

            if elapsed > max_wait_time:
                raise VersaStatError("could not start experiment")
                raise KeyboardInterrupt("could not start.")
                break

        print("started experiment sequence successfully.")

        return

    def stop(self):
        """ Stops the sequence of actions that is currently running in the instrument that is currently connected. """
        self.instrument.Experiment.Stop()

    def skip(self):
        """Skips the currently running action and immediately starts the next action.
        If there is no more actions to run, the sequence is simply stopped.
        """
        self.instrument.Experiment.Skip()

    def sequence_running(self):
        """ Returns true if a sequence is currently running on the connected instrument, false if not. """
        return self.instrument.Experiment.IsSequenceRunning()

    def points_available(self):
        """Returns the number of points that have been stored by the instrument after a sequence of actions has begun.
        Returns -1 when all data has been retrieved from the instrument.
        """
        return self.instrument.Experiment.GetNumPointsAvailable()

    def last_open_circuit(self):
        """Returns the last measured Open Circuit value.
        This value is stored at the beginning of the sequence (and updated anytime the “AddMeasureOpenCircuit” action is called)"""
        return self.instrument.Experiment.GetLastMeasuredOC()

    # The following Action Methods can be called in order to create a sequence of Actions.
    # A single string argument encodes multiple parameters as comma-separated lists...
    # For example, AddOpenCircuit( string ) could be called, then AddEISPotentiostatic( string ) called.
    # This would create a sequence of two actions, when started, the open circuit experiment would run, then the impedance experiment.

    # TODO: write a class interface for different experimental actions to streamline logging and serialization?

    # TODO: code-generation for GetData* interface?

    def potential(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()
            num_points = num_points - start

        values = self.instrument.Experiment.GetDataPotential(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def current(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()
            num_points = num_points - start

        values = self.instrument.Experiment.GetDataCurrent(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def elapsed_time(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()
            num_points = num_points - start

        values = self.instrument.Experiment.GetDataElapsedTime(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def applied_potential(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()
            num_points = num_points - start

        values = self.instrument.Experiment.GetDataAppliedPotential(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def segment(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()
            num_points = num_points - start

        values = self.instrument.Experiment.GetDataSegment(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def current_range_history(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()
            num_points = num_points - start

        values = self.instrument.Experiment.GetDataCurrentRange(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def frequency(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()
            num_points = num_points - start

        values = self.instrument.Experiment.GetDataFrequency(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def impedance_real(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()
            num_points = num_points - start

        values = self.instrument.Experiment.GetDataImpedanceReal(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def impedance_imag(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()
            num_points = num_points - start

        values = self.instrument.Experiment.GetDataImpedanceImaginary(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def hardcoded_open_circuit(self, params):
        default_params = (
            "1,10,NONE,<,0,NONE,<,0,2MA,AUTO,AUTO,AUTO,INTERNAL,AUTO,AUTO,AUTO"
        )
        status = self.instrument.Experiment.AddOpenCircuit(default_params)
        return status, default_params

    def measure_open_circuit(self):
        status = self.instrument.Experiment.AddMeasureOpenCircuit()
        return status, None
