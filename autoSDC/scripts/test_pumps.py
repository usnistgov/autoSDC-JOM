import sys
import numpy as np

sys.path.append(".")
from asdc import sdc


def test_pump_array():
    print("connecting to pumps...")
    p = sdc.pump.PumpArray(port="COM6")
    p.print_config()


if __name__ == "__main__":
    test_pump_array()
