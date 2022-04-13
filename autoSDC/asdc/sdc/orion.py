import os
import re
import time
import serial
import typing
import asyncio
import streamz
import argparse
import threading
import collections
import pandas as pd
from datetime import datetime
from contextlib import contextmanager

from typing import Dict, List

import zmq
import zmq.asyncio

MODEL_NUMBER = "A221"

# the A221 meter responds to GETMEAS with a comma-delimited string
# RESPONSE_COLUMNS contains field names for this string
# LOGFILE_COLUMNS specifies the columns that we explicitly want to save
RESPONSE_COLUMNS = (
    "model",
    "serialnum",
    "software_ver",
    "userid",
    "timestamp",
    "sample_id",
    "channel",
    "mode",
    "pH",
    "pH_unit",
    "mV",
    "mV_unit",
    "temperature",
    "temperature_unit",
    "slope",
    "slope_unit",
    "methodnum",
    "lognum",
)

LOGFILE_COLUMNS = (
    "timestamp",
    "pH",
    "pH_unit",
    "mV",
    "mV_unit",
    "temperature",
    "temperature_unit",
    "slope",
    "slope_unit",
)

# set up and bind zmq publisher socket
DASHBOARD_PORT = 2345
DASHBOARD_ADDRESS = "127.0.0.1"
DASHBOARD_URI = f"tcp://{DASHBOARD_ADDRESS}:{DASHBOARD_PORT}"


def encode(message: str) -> bytes:
    message = message + "\r"
    return message.encode()


class PHMeter:

    supported_modes = {"pH", "mV"}

    def __init__(
        self,
        address,
        baud=19200,
        timeout=2,
        mode="pH",
        model_number=MODEL_NUMBER,
        buffer_size=64,
        zmq_pub=False,
    ):
        self.model_number = MODEL_NUMBER

        self.pH = collections.deque(maxlen=buffer_size)
        self.temperature = collections.deque(maxlen=buffer_size)

        self.ser = serial.Serial(
            port=address,
            baudrate=baud,
            parity=serial.PARITY_NONE,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )

        self.mode = mode
        self.timeout = timeout
        self._blocking = False
        self.blocking = False

        if zmq_pub:
            self.context = zmq.asyncio.Context.instance()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(DASHBOARD_URI)
        else:
            self.socket = None

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> bool:
        if mode in self.supported_modes:
            self._mode = mode
        else:
            raise ValueError(f"meter mode must be one of {self.supported_modes}")

        self.write(f"SETMODE {self.mode}")
        return self.check_response()

    @property
    def blocking(self) -> bool:
        return self._blocking

    @blocking.setter
    def blocking(self, blocking: bool):
        assert isinstance(blocking, bool)

        if self._blocking == blocking:
            pass
        elif blocking:
            self.ser.timeout = None
        elif not blocking:
            self.ser.timeout = self.timeout

        self._blocking = blocking

    @contextmanager
    def sync(self):
        try:
            self.blocking = True
            yield
        finally:
            self.blocking = False

    def check_response(self) -> bool:
        response = self.ser.read(size=2).decode()
        print(f"response: {response}")
        if response == "> ":
            return True
        else:
            return False

    def set_csv(self):
        self.write("SETCSV")
        return self.check_response()

    def write(self, msg: str) -> None:
        self.ser.write(encode(msg))

    @staticmethod
    def _to_dict(data: str) -> Dict[str, float]:
        values = data.split(",")
        d = dict(pH=float(values[1]), temperature=float(values[5]))
        return d

    def _process_pH(self, response: str, timestamp: datetime) -> str:
        """
        Meter Model, Serial Number, Software Revision, User ID, Date & Time, Sample ID, Channel, Mode, pH Value, pH Unit, mV Value, mV Unit, Temperature Value, Temperature Unit, Slope Value, Slope Unit, Method #, Log #

        b'GETMEAS         \r\n\r\n\rA221 pH,K10231,3.04,ABCDE,08/26/20 00:20:59,---,CH-1,pH,6.92,pH,-3.8, mV,23.3,C,99.6,%,M100,#162\n\r\r>'

        TODO: get timestamp, sample_id, channel, mode, pH_value, pH_unit, mV_value, mV_unit, temperature_value, temperature_unit, slope_value, slope_unit
        """
        # index into the response by searching for the model number
        model_match = re.search(self.model_number, response)
        data_idx = model_match.start()

        # remove any >, \r, and \n characters
        data = re.sub("[\>\\r\\n]", "", response[data_idx:])

        # strip any whitespace adjacent to the comma delimiters
        # and reconstruct the CSV string
        values = list(map(str.strip, data.split(",")))

        # coerce into pandas dataframe to select columns
        row = pd.DataFrame([values], columns=RESPONSE_COLUMNS)
        v = row.loc[:, LOGFILE_COLUMNS]
        v["timestamp"] = timestamp
        data = v.to_csv(header=False, index=False).strip()

        return data

    def read(self, count: int = 1):

        if count == 1:
            self.write("GETMEAS")
            response = self.ser.read_until(b">")
            timestamp = datetime.now()
            # print(response)

            data = self._process_pH(response.decode(), timestamp)
            return data

        elif count > 1:
            # FIXME: the multi-reading `GETMEAS` command will only return
            # a single `>` at the end of all the response lines...
            self.write(f"GETMEAS {count}")
            responses = [self.ser.read_until(b">") for response in range(count)]
            data = [self._process_pH(response.decode()) for response in responses]
            return data

    def readloop(self, stop_event=None, interval=30, logfile="pHmeter_test.csv"):

        # start an asyncio event loop in the worker thread...
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # clear the output buffer...
        buf = self.ser.read(500)

        start_ts = pd.Timestamp(datetime.now())

        def update_buffers(values):
            # push values into deques for external monitoring...
            self.pH.append(values["pH"])
            self.temperature.append(values["temperature"])

        # skip writing header line if appending to existing logfile
        write_header = True
        if os.path.isfile(logfile):
            write_header = False

        with self.sync():
            with open(logfile, "a") as f:

                if write_header:
                    # write a CSV header if the file is new
                    print(",".join(LOGFILE_COLUMNS), file=f)

                source = streamz.Source()
                log = source.sink(lambda x: print(x, file=f))
                values = source.map(self._to_dict)
                buf = values.sink(update_buffers)

                if self.socket is not None:
                    start = pd.Timestamp(datetime.now())
                    reltime = lambda: pd.Timestamp(datetime.now()) - start
                    df = values.map(
                        lambda x: pd.DataFrame(x, index=[reltime().total_seconds()])
                    )
                    df.sink(lambda x: self.socket.send_pyobj(x))

                # main measurement loop to run at interval
                while not stop_event.is_set():

                    target_ts = time.time() + interval

                    reading = self.read()
                    source.emit(reading)

                    # wait out the rest of the interval
                    # but continue immediately if this iteration took longer than the interval
                    # return immediately if signalled
                    stop_event.wait(timeout=max(0, target_ts - time.time()))

    @contextmanager
    def monitor(self, interval=30, logfile="pHmeter_test.csv"):
        """use this like

        ```
        with meter.monitor(interval=10):
            time.sleep(60)
        ```
        """
        stop_event = threading.Event()

        io_worker = threading.Thread(
            target=self.readloop, args=(stop_event, interval, logfile)
        )
        try:
            io_worker.start()
            yield
        finally:
            stop_event.set()
            io_worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pH meter client")
    parser.add_argument("--port", default="COM21", help="COM port for the pH meter")
    parser.add_argument(
        "--verbose", action="store_true", help="include extra debugging output"
    )
    args = parser.parse_args()

    meter = PHMeter(args.port)

    with meter.monitor(interval=10):
        time.sleep(240)
        print(meter.pH)
