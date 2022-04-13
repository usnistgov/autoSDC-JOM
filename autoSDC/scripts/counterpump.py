import sys

sys.path.append(".")
from asdc.sdc import microcontroller

adafruit_port = "COM9"
p = microcontroller.PeristalticPump(port=adafruit_port, timeout=1)

# parse commandline arguments
args = sys.argv

if len(args) == 2:
    setpoint = args[1]
    try:
        setpoint = float(setpoint)
        if setpoint < 0 or setpoint > 1:
            raise ValueError
    except ValueError:
        if setpoint != "stop":
            raise ("setpoint must be between 0 and 1")
elif len(args) == 1:
    print("starting the pump with default flow rate")
    setpoint = 0.3

if setpoint == "stop":
    print(f"stopping the pump!")
    p.stop()
else:
    print(f"setting flow rate to {setpoint}")
    p.set_flow_proportion(setpoint)
    p.start()
