import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData

logger = logging.getLogger(__name__)


class PotentiodynamicData(EchemData):
    @property
    def _constructor(self):
        return PotentiodynamicData

    @property
    def name(self):
        return "Potentiodynamic"

    def check_quality(self):
        """ Potentiodynamic heuristics: not implemented. """
        return True

    def plot(self, fit=False):
        """ plot Potentiodynamic: current vs potential """
        # # super().plot('current', 'potential')
        plt.plot(self["potential"], self["current"])
        plt.xlabel("potential (V)")
        plt.ylabel("current (A)")

        plt.tight_layout()
