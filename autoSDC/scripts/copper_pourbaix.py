import sys
import click
import numpy as np
from ruamel import yaml

sys.path.append(".")
from asdc import sdc


@click.command()
@click.argument("config-file", type=click.Path())
def copper_pourbaix(config_file):

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    solutions = config.get("solutions")

    print("connecting to pumps...")
    p = sdc.pump.PumpArray(solutions, port="COM6")
    p.print_config()

    p.run_all()

    for setpoint in [2.0, 3.0, 4.0, 5.0]:
        print("setpoint pH:", setpoint)
        p.set_pH(setpoint=setpoint)

        input("Press enter to continue to the next pH setpoint...")

    p.stop_all()


if __name__ == "__main__":
    copper_pourbaix()
