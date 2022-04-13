import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData

logger = logging.getLogger(__name__)


class LSVData(EchemData):
    @property
    def _constructor(self):
        return LSVData

    @property
    def name(self):
        return "LSV"

    def check_quality(self):
        """ LSV heuristics: not implemented. """
        return True

    def plot(self, fit=False):
        """ plot LSV: current vs potential """
        # # super().plot('current', 'potential')
        plt.plot(self["potential"], self["current"])
        plt.xlabel("potential (V)")
        plt.ylabel("current (A)")

        plt.tight_layout()
