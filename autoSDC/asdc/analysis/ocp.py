import logging
import numpy as np
import pandas as pd
from csaps import csaps
from pandas import DataFrame
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData

logger = logging.getLogger(__name__)


def ocp_stop(x, y, time=90, tstart=300, thresh=0.00003):
    t = tstart
    deriv = np.inf
    while (np.abs(deriv) > thresh) and (t < 900):
        tstop = t + time
        deriv = np.mean(y[(x > t) & (x < tstop)])
        t = tstop
    return t


def ocp_convergence(ocp, smooth=0.001, tr=100):
    """model an open circuit potential trace to check that it converges to a constant value

    computes the average slope at the end of the potential trace
    the RMS error of a cubic spline model
    the maximum potential jump over a single measurement interval
    and the hold stop time
    """
    t, potential = ocp["elapsed_time"], ocp["potential"]

    # estimate the derivative using a cubic spline model
    model = csaps(t, potential, smooth=smooth)
    dVdt = model.spline.derivative()(t)

    tstop = ocp_stop(t, dVdt)

    # average the smoothed derivative over the last time chunk
    checktime = t.max() - tr
    avslope = np.mean(dVdt[t > checktime])

    # compute the largest spike in the finite difference derivative
    maxdiff = np.max(np.abs(np.diff(potential)))

    # compute the root-mean-square error of the spline model
    rms = np.sqrt(np.mean((model.spline(t) - potential) ** 2))

    results = {
        "average_slope": avslope,
        "rms": rms,
        "spike": maxdiff,
        "stop time": tstop,
    }

    return results


class OCPData(EchemData):
    @property
    def _constructor(self):
        return OCPData

    @property
    def name(self):
        return "OCP"

    def check_quality(self):
        """ OCP convergence criteria """
        convergence_stats = ocp_convergence(self)

        if convergence_stats["spike"] > 0.1:
            logger.warning("OCP potential trace failed smoothness heuristic.")

        logger.info(f"OCP check: {convergence_stats}")

        return convergence_stats

    def plot(self):
        """ plot open circuit potential vs elapsed time """
        # super().plot('elapsed_time', 'potential')
        plt.plot(self.elapsed_time, self.potential)
        plt.xlabel("elapsed time (s)")
        plt.ylabel("potential (V)")
        plt.tight_layout()
