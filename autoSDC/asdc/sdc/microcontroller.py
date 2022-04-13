import json
import time
import serial

from typing import Optional, Dict

from asdc.sdc.utils import encode, decode
from asdc.sdc.utils import flow_to_proportion, proportion_to_flow


class MicrocontrollerInterface:
    """base interface for the equipment hooked up through a microcontroller board

    This interface currently uses the [ndjson](https://github.com/ndjson/ndjson-spec) protocol to send commands to the board.
    """

    def __init__(
        self, port: str = "COM9", baudrate: int = 115200, timeout: float = 0.5
    ):
        """Microcontroller interface

        Arguments:
            port: serial port for the microcontroller board
            baudrate: serial protocol timing
            timeout: default serial interface timeout period (s)
        """

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

    def eval(self, command: Dict, timeout: Optional[float] = None) -> str:
        """Microcontroller interface

        open a serial connection to the microcontroller, send an `ndjson` command,
        and fetch and return the result from the board.

        Arguments:
            command: serial port for the microcontroller board
            timeout: override default serial interface timeout period (s)


        Returns:
            response from the board. currently the board does not strictly
            send `ndjson` responses.
        """

        if timeout is None:
            timeout = self.timeout

        with serial.Serial(
            port=self.port, baudrate=self.baudrate, timeout=timeout
        ) as ser:

            # block until the whole command is echoed
            ser.write(encode(json.dumps(command, separators=(",", ":"))))
            ack = ser.readline()
            print(decode(ack))

            # block until an ndjson response is received
            response = ser.readline()
            return decode(response)

    def read(self):
        with serial.Serial(
            port=self.port, baudrate=self.baudrate, timeout=self.timeout
        ) as ser:
            response = ser.readlines()
            return decode(response)


class Reflectometer(MicrocontrollerInterface):
    """ microcontroller interface for the [ThorLabs PDA36A2](https://www.thorlabs.com/thorproduct.cfm?partnumber=PDA36A2) """

    def collect(self, timeout: Optional[float] = None) -> float:
        """collect reading from laser reflectance setup

        Arguments:
            timeout: override default serial interface timeout period (s)

        Returns:
            average voltage reading from the photodiode, corresponding to the sample reflectance.
            the average is taken over 25 readings (with a 50 ms interval between readings)
        """

        if timeout is None:
            timout = self.timeout

        response = self.eval({"op": "laser"}, timeout=timeout)

        # TODO: check response content / add response status info
        # reflectance_data = json.loads(response[1])
        reflectance = float(response)

        return reflectance


class PeristalticPump(MicrocontrollerInterface):
    """ microcontroller interface for the [ISMATEC peristaltic pump](http://www.ismatec.com/images/pdf/manuals/IP.pdf) """

    def start(self):
        """ start pumping """
        return self.eval({"op": "start"})

    def stop(self):
        """ start pumping """
        return self.eval({"op": "stop"})

    def get_flow(self) -> float:
        """ get voltage encoding flow rate from the board... """
        return self.eval({"op": "get_flow"})

    def set_flow(self, rate: float):
        """set pumping rate to counterbalance a nominal target flow rate in mL/min

        This uses a rough calibration curve defined in [asdc.sdc.utils.flow_to_proportion][]

        Arguments:
            rate: nominal flow rate in mL/min
        """

        ismatec_proportion = flow_to_proportion(rate)
        print(f"ismatec_proportion: {ismatec_proportion}")
        self.eval({"op": "set_flow", "rate": ismatec_proportion})

    def set_flow_proportion(self, proportion: float):
        """set proportional flow rate

        Arguments:
            proportion: nominal flow rate as a fraction of the pump capacity in `(0, 1)`

        """
        self.eval({"op": "set_flow", "rate": proportion})


class Light(MicrocontrollerInterface):
    """ microcontroller interface for the light (for illumination during optical characterization """

    def set(self, value):
        """ set the light state to on/off """
        if value in ("on", "off"):
            self.eval({"op": "light", "value": value})
        else:
            raise ValueError

    def on(self):
        """ turn the light on """
        self.set("on")

    def off(self):
        """ turn the light off """
        self.set("off")
