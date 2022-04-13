import sys

sys.path.append(".")

import json
import time
import numpy as np
import pandas as pd
from scipy import stats

from asdc.sdc import utils
from asdc.sdc import microcontroller

pump = microcontroller.PeristalticPump()

proportion = np.linspace(0, 0.6, 0.05)
volts_in, volts_out, sem_volts_out = [], [], []

pump.start()

for p in proportion:
    volts_in.append(p * 3.3)
    pump.set_flow_proportion(p)

    v = []
    for iteration in range(5):
        time.sleep(1)
        v.append(pump.get_flow())
    volts_out.append(np.mean(v))
    sem_volts_out.append(stats.sem(v))

pump.stop()

df = pd.DataFrame(
    {
        "proportion": proportion,
        "volts_in": volts_in,
        "volts_out": volts_out,
        "sem_volts_out": sem_volts_out,
    }
)

df.to_csv("ismatec_calib.csv")
