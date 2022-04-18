import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData

logger = logging.getLogger(__name__)


class PotentiostaticData(EchemData):
    @property
    def _constructor(self):
        return PotentiostaticData

    @property
    def name(self):
        return "Potentiostatic"

    def check_quality(self):
        """ Potentiostatic heuristics: not implemented. """
        return True

    def plot(self, fit=False):
        """ plot Potentiostatic: current vs potential """
        # # super().plot('current', 'potential')
        plt.plot(self["elapsed_time"], self["current"])
        plt.xlabel("elapsed time (s)")
        plt.ylabel("current (A)")

        plt.tight_layout()
