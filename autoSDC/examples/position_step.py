#!/usr/bin/env python

import asdc.position


def position_step(speed=1e-5):

    with asdc.position.controller(ip="192.168.10.11", speed=speed) as pos:

        pos.print_status()
        pos.update_x(delta=1e-4, verbose=True)
        pos.print_status()


if __name__ == "__main__":
    position_step()
