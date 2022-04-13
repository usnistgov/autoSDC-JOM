from __future__ import annotations

import time
import chempy
import serial
import logging
import numpy as np
from scipy import optimize
from chempy import equilibria
from collections import defaultdict
from collections.abc import Iterable

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from asdc.sdc.utils import encode
from asdc.sdc.microcontroller import PeristalticPump

# placeholder config for development
SOLUTIONS = {0: {"H2SO4": 0.1}, 1: {"Na2SO4": 0.1}, 2: {"CuSO4": 0.1}}


def mix(solutions, fraction):
    """ compute nominal compositions when mixing multiple solutions """

    solution = defaultdict(float)
    for sol, x in zip(solutions.values(), fraction):
        for species, conc in sol.items():
            solution[species] += x * conc
    return solution


def sulfuric_eq_pH(solution, verbose=False):

    eqsys = equilibria.EqSystem.from_string(
        """
        HSO4- = H+ + SO4-2; 10**-2
        H2SO4 = H+ + HSO4-; 2.4e6
        H2O = H+ + OH-; 10**-14/55.4
        """
    )

    nominal_sulfates = solution["CuSO4"] + solution["Na2SO4"]
    arr, info, sane = eqsys.root(
        defaultdict(
            float, {"H2O": 55.4, "H2SO4": solution["H2SO4"], "SO4-2": nominal_sulfates}
        )
    )
    conc = dict(zip(eqsys.substances, arr))

    pH = -np.log10(conc["H+"])

    if verbose:
        print("pH: %.2f" % pH)
        print()
        pprint(conc)

    return -np.log10(conc["H+"])


def pH_error(target_pH, stock=SOLUTIONS):
    def f(x):
        """ perform linear mixing between just two solutions """
        s = mix(stock, [x, 1 - x, 0])
        pH = sulfuric_eq_pH(s, verbose=False)
        return pH

    return lambda x: f(x) - target_pH


class PumpArray:
    """ KDS Legato pump array interface """

    def __init__(
        self,
        solutions=SOLUTIONS,
        diameter=29.5,
        port="COM7",
        baud=115200,
        timeout=1,
        output_buffer=100,
        fast=False,
        flow_rate=0.5,
        flow_units="ml/min",
    ):
        """pump array.
        What is needed? concentrations and flow rates.
        Low level interface: set individual flow rates
        High level interface: set total flow rate and composition

        TODO: look into using serial.tools.list_ports.comports to identify the correct COM port to connect to...
        the id string should be something like 'USB serial port for Syringe Pump (COM*)'
        """
        self.solutions = solutions
        self.syringe_diameter = diameter

        # serial interface things
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.buffer_size = output_buffer
        self.fast = fast
        self.flow_rate = flow_rate
        self.flow_units = flow_units
        self.flow_setpoint = {pump_id: 0.0 for pump_id in self.solutions.keys()}

        # pump initialization
        # self.diameter(self.syringe_diameter)

    def relative_rates(self):
        total_rate = sum(self.flow_setpoint.values())
        return {key: rate / total_rate for key, rate in self.flow_setpoint.items()}

    def eval(self, command, pump_id=0, ser=None, check_response=False, fast=False):
        """evaluate a PumpChain command.
        consider batches commands together using connection `ser`
        """

        if fast or self.fast:
            command = "@{}".format(command)

        # TODO: run a command for every configured pump if pump_id is None
        # if isinstance(pump_id, Iterable):
        #     for id in pump_id:
        #         self.eval()

        command = f"{pump_id} {command}"

        if ser is not None:
            ser.write(encode(command))

            if check_response:
                s = ser.read(self.buffer_size)
                return s
        else:
            with serial.Serial(
                port=self.port, baudrate=self.baud, timeout=self.timeout
            ) as ser:
                ser.write(encode(command))
                if check_response:
                    s = ser.read(self.buffer_size)
                    return s

    def refresh_ui(self, pump_id=0):
        """ for whatever reason, 'ver' refreshes the pump UI when other commands do not """
        self.eval("ver", pump_id=pump_id)

    def run(self, pump_id=0):
        print(f"asking pump {pump_id} to run")
        self.eval("run", pump_id=pump_id)

    def run_all(self, fast=False):
        with serial.Serial(
            port=self.port, baudrate=self.baud, timeout=self.timeout
        ) as ser:
            for pump_id in self.solutions.keys():
                if self.flow_setpoint[pump_id] > 0:
                    self.eval("run", pump_id=pump_id, ser=ser, fast=fast)
                    time.sleep(0.25)
                else:
                    self.eval("stop", pump_id=pump_id, ser=ser, fast=fast)
                    time.sleep(0.25)

    def refresh_all(self):
        with serial.Serial(
            port=self.port, baudrate=self.baud, timeout=self.timeout
        ) as ser:
            for pump_id in self.solutions.keys():
                self.eval("ver", pump_id=pump_id, ser=ser)
                time.sleep(0.05)

    def stop(self, pump_id=0):
        self.eval("stop", pump_id=pump_id)

    def stop_all(self, fast=False):

        with serial.Serial(
            port=self.port, baudrate=self.baud, timeout=self.timeout
        ) as ser:
            for pump_id in self.solutions.keys():
                self.eval("stop", pump_id=pump_id, ser=ser, fast=fast)
                time.sleep(0.25)

    def diameter(self, setpoint=None, pump_id=None):

        if setpoint is not None:
            command = "diameter {}".format(setpoint)
        else:
            command = "diameter"

        if pump_id is None:
            with serial.Serial(
                port=self.port, baudrate=self.baud, timeout=self.timeout
            ) as ser:
                for id in self.solutions.keys():
                    self.eval(
                        command, pump_id=id, ser=ser, fast=True, check_response=True
                    )
        else:
            self.eval(command, pump_id=pump_id)

    def infusion_rate(self, ser=None, pump_id=0, rate=None, units="ml/min", fast=False):

        if rate is not None:
            command = "irate {} {}".format(rate, units)
        else:
            command = "irate"

        self.eval(command, pump_id=pump_id, ser=ser, fast=fast)

    def set_pH(self, setpoint=3.0):
        """ control pH -- limited to two pumps for now. """

        if setpoint == 7.0:
            print("forcing Na2SO4-only run")
            x = 0.0
        else:
            x, r = optimize.brentq(
                pH_error(setpoint, stock=self.solutions), 0, 1, full_output=True
            )

        print(x)

        self.infusion_rate(pump_id=0, rate=x * self.flow_rate, units=self.flow_units)
        self.infusion_rate(
            pump_id=1, rate=(1 - x) * self.flow_rate, units=self.flow_units
        )

        self.flow_setpoint = {0: x * self.flow_rate, 1: (1 - x) * self.flow_rate}

    def get_pump_id(self, q):
        for key, value in self.solutions.items():
            if q in value:
                return key

    def levels(self):
        """check syringe levels to ensure enough solution is available for a given push

        for each pump, run IVOLUME and TVOLUME to compute the remaining solution level

        Note: does not handle the case where the target setpoint is reached and the pump stops!
              This function is meant to be used preemptively to avoid this happening in the first place.
        """

        def decode_level(response):
            """parse *volume response and convert to mL

            expect something like `b'\n00:7.01337 ul\r\n00:'`

            the main response being {address:02d}:{volume:f} {unit}
            """
            response = response.decode().strip().split("\r\n")
            (
                response,
                *prompt,
            ) = response  # unpack response and hope the real response is the first line
            address, level = response.split(":")

            # expect something like level == '7.01337 ul'
            level, unit = level.split()
            level = float(level)

            # somewhat hacky unit conversions to mL
            if unit == "ml":
                pass
            elif unit == "ul":
                level *= 1e-3
            elif unit == "nl":
                level *= 1e-6

            return level

        volume_remaining = {}
        with serial.Serial(
            port=self.port, baudrate=self.baud, timeout=self.timeout
        ) as ser:
            for pump_id, solution in self.solutions.items():

                # TODO: solutions need better names
                name = list(solution.keys())[0]

                r = self.eval(
                    "tvolume", pump_id=pump_id, check_response=True, fast=True, ser=ser
                )
                print(r.decode())
                logger.debug(f"tvolume: {r.decode()}")
                target_volume = decode_level(r)

                r = self.eval(
                    "ivolume", pump_id=pump_id, check_response=True, fast=True, ser=ser
                )
                print(r.decode())
                logger.debug(f"ivolume: {r.decode()}")
                infused_volume = decode_level(r)

                # print(f'tvolume: {target_volume} mL')
                # print(f'ivolume: {infused_volume} mL')
                # print(f'remaining: {target_volume - infused_volume} mL')
                volume_remaining[name] = target_volume - infused_volume

        return volume_remaining

    def set_rates(self, setpoints, units="ml/min", start=False, fast=False):
        """directly set absolute flow rates

        flow_setpoint is a dict containing absolute flow rates for each syringe
        TODO: incorporate peristaltic pump here and set rates appropriately? need to set rates separately sometimes.
        """

        total_setpoint = sum(setpoints.values())
        print(f"total_setpoint: {total_setpoint}")

        # reset rates to 0
        for pump_id in self.flow_setpoint.keys():
            self.flow_setpoint[pump_id] = 0.0

        # set flowrates for the syringe pump array
        with serial.Serial(
            port=self.port, baudrate=self.baud, timeout=self.timeout
        ) as ser:
            print(setpoints)
            time.sleep(0.05)
            for species, setpoint in setpoints.items():
                print(species, setpoint)
                pump_id = self.get_pump_id(species)
                print(pump_id)
                if setpoint > 0:
                    self.flow_setpoint[pump_id] = setpoint
                    self.infusion_rate(
                        pump_id=pump_id, ser=ser, rate=setpoint, units=units, fast=fast
                    )
                    time.sleep(0.25)

        time.sleep(0.25)

        if start:
            self.run_all()

        print(self.flow_setpoint)
