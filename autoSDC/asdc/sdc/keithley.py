"""Keithley 2450 interface -- extends qcodes Keithley2450."""
import json
import time

import numpy as np
import pandas as pd
from datetime import datetime
from qcodes.instrument_drivers.tektronix.Keithley_2450 import Keithley2450

# experimentally-determined measurement overhead
# per power line cycle
OVERHEAD = 0.069


class Keithley(Keithley2450):
    def __init__(
        self,
        name="keithley",
        address="USB0::0x05E6::0x2450::04461816::INSTR",
        timeout=5,
    ):
        """Extend qcodes Keithley2450 interface."""
        super().__init__(name=name, address=address, timeout=timeout)
        self.reset()

    def run(self, experiment):
        """Just wrap around constant current for now"""

        argstring = experiment.getargs()
        metadata = {"timestamp_start": datetime.now(), "parameters": argstring}

        results = self.constant_current(
            experiment.current, experiment.duration, experiment.interval
        )

        metadata["timestamp_end"] = datetime.now()
        metadata["error_codes"] = json.dumps([])

        # cast to EChemData type
        results = experiment.marshal(results)

        return results, metadata

    def constant_current(self, setpoint: float, duration: float, interval: float):
        """Apply constant current `setpoint` (A) for `duration` (s).
        Record potential with every `interval` (s)
        """

        # measure voltage while controlling current
        self.sense_function("voltage")
        self.source.function("current")

        # configure user data acquisition
        buffer_name = "userbuff1"
        buffer_size = 5000

        # power line cycles per measurement
        nplc = float(self.sense.get("nplc"))

        delay = interval - nplc * OVERHEAD

        n_points = np.ceil(duration / interval)
        total_time = n_points * delay

        # sync current source time with host system clock
        t = datetime.now()
        self.write(
            f"syst:time {t.year}, {t.month}, {t.day}, {t.hour}, {t.minute}, {t.second}"
        )

        elems = ["source_value", "measurement", "relative_time", "time", "date"]
        with self.buffer(buffer_name, buffer_size) as buffer:
            buffer.elements(elems)
            self.source.sweep_setup(
                setpoint,
                setpoint,
                n_points,
                buffer_name=buffer.buffer_name,
                delay=delay,
                fail_abort="OFF",
            )

            timeout_old = self.timeout()
            self.timeout(total_time + 60)
            data = self.sense.sweep()

            all_data = self.sense.sweep.get_selected()
            # revert timeout
            self.timeout(timeout_old)

        # parse data stream to pandas array
        df = pd.DataFrame(np.array(all_data).reshape(-1, len(elems)), columns=elems)
        df["measurement"] = pd.to_numeric(df["measurement"])
        df["source_value"] = pd.to_numeric(df["source_value"])
        df["relative_time"] = pd.to_numeric(df["relative_time"]) + delay
        df.rename(
            columns={
                "relative_time": "elapsed_time",
                "source_value": "current",
                "measurement": "potential",
            },
            inplace=True,
        )

        return df
