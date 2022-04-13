import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData

logger = logging.getLogger(__name__)


class CVData(EchemData):
    @property
    def _constructor(self):
        return CVData

    @property
    def name(self):
        return "CV"

    def check_quality(self):
        """ CV heuristics: not implemented. """
        return True

    def plot(self, fit=False):
        """ plot CV: current vs potential """
        # # super().plot('current', 'potential')
        plt.plot(self["potential"], self["current"])
        plt.xlabel("potential (V)")
        plt.ylabel("current (A)")

        plt.tight_layout()
