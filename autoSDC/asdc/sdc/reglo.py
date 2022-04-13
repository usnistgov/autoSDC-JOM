""" interface for the Reglo peristaltic pump """

import time
import typing
from typing import Dict
from enum import IntEnum
from collections import Iterable

import regloicclib


class Channel(IntEnum):
    """index organization for the pump channels

    needle defaults counterclockwise (-)
    drain (-)
    loop (+)
    rinse (+)
    """

    ALL = 0
    NEEDLE = 1
    DRAIN = 2
    LOOP = 3
    RINSE = 4


# 12 mL/min

# order SOURCE, LOOP, DRAIN
CHANNEL_UPDATE_DELAY = 0.001


class Reglo(regloicclib.Pump):
    """thin wrapper around the pump interface from regloicc

    TODO: rewrite the serial interface...
    """

    def __init__(self, address=None, debug=False, tubing_inner_diameter=1.52):
        super().__init__(address=address, debug=debug)
        self.tubing_inner_diameter = tubing_inner_diameter

        # TODO: this should maybe be a python property
        # so that max flow rates can be automatically kept in sync
        for channel in range(1, 5):
            self.setTubingInnerDiameter(self.tubing_inner_diameter, channel=channel)

        self.maxrates = {}
        for channel in self.channels:
            self.maxrates[channel] = float(self.hw.query("%d?" % channel).split(" ")[0])
            time.sleep(CHANNEL_UPDATE_DELAY)

    def set_rates(self, setpoints: Dict[Channel, float]):

        for channel, rate in setpoints.items():
            if rate == 0:
                self.stop(channel=channel.value)
            else:
                self.continuousFlow(rate, channel=channel.value)
            time.sleep(CHANNEL_UPDATE_DELAY)

        return

    def continuousFlow(self, rate, channel=None):
        """
        Start continuous flow at rate (ml/min) on specified channel or
        on all channels.
        """

        if type(channel) is Channel:
            channel = channel.value

        if channel is None or channel == 0:
            channel = 0
            # this enables fairly synchronous start
            maxrate = min(self.maxrates.values())
        else:
            maxrate = self.maxrates[channel]

        assert channel in self.channels or channel == 0

        # flow rate mode
        self.hw.write("%dM" % channel)
        self.hw.write(f"{channel}M")

        # set flow direction
        if rate < 0:
            self.hw.write(f"{channel}K")
        else:
            self.hw.write(f"{channel}J")

        # set flow rate
        if abs(rate) > maxrate:
            rate = rate / abs(rate) * maxrate

        flowrate = self._volume2(rate)
        self.hw.query(f"{channel}f{flowrate}")

        # maintain internal running status in python client
        self.hw.setRunningStatus(True, channel)

        # start pumps command
        self.hw.write(f"{channel}H")

    def stop(self, channel=None):

        if channel is None or type(channel) is int:
            super().stop(channel=channel)

        elif type(channel) is Channel:
            super().stop(channel=channel.value)

        elif isinstance(channel, Iterable):
            for c in channel:
                super().stop(channel=c.value)

        return
