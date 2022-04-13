import sys
import numpy as np

sys.path.append(".")
from asdc import sdc

solutions = {0: {"H2SO4": 1.0}, 1: {"Na2SO4": 1.0}, 2: {"KOH": 1.0}}


def test_pump_array():
    print("connecting to pumps...")
    p = sdc.pump.PumpArray(solutions, port="COM6")
    p.print_config()

    # p.set_pH(setpoint=2.0)
    p.set_rates({"KOH": 1.0, "H2SO4": 0.5})


if __name__ == "__main__":
    test_pump_array()
