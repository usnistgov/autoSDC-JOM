import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData

logger = logging.getLogger(__name__)


class ConstantCurrentData(EchemData):
    @property
    def _constructor(self):
        return ConstantCurrentData

    @property
    def name(self):
        return "ConstantCurrent"

    def check_quality(self):
        """ Potentiostatic heuristics: not implemented. """
        return True

    def plot(self, fit=False):
        """ plot Constant Current:  potential vs time """
        # # super().plot('current', 'potential')
        plt.plot(self["elapsed_time"], self["potential"])
        plt.xlabel("elapsed time (s)")
        plt.ylabel("potential (V)")

        plt.tight_layout()
