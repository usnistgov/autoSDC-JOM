import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData

logger = logging.getLogger(__name__)


class PotentiostaticEISData(EchemData):
    @property
    def _constructor(self):
        return PotentiostaticEISData

    @property
    def name(self):
        return "PotentiostaticEIS"

    def check_quality(self):
        """ Potentiostatic EIS heuristics: not implemented. """
        return True

    def plot(self, fit=False):
        """ EIS Bode plot: |Z| and phase(Z) vs log(ω) """
        fig, axes = plt.subplots(nrows=2, sharex=True)
        plt.sca(axes[0])

        # construct complex impedance
        Z = self["impedance_real"] + 1j * self["impedance_imag"]

        plt.plot(self["frequency"], np.abs(Z))
        plt.semilogy()
        plt.ylabel("|Z|")

        plt.sca(axes[1])
        plt.plot(self["frequency"], np.degrees(np.angle(Z)))
        plt.ylabel("phase(Z)")

        plt.xlabel("frequency (Hz)")

        plt.semilogx()

        plt.tight_layout()
