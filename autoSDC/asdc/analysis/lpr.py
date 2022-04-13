from __future__ import annotations

import lmfit
import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from asdc.analysis.echem_data import EchemData, Status

logger = logging.getLogger(__name__)


def current_crosses_zero(df: pd.DataFrame) -> bool:
    """ verify that a valid LPR scan has a current trace that crosses zero """
    current = df["current"]
    logger.debug("LPR check")
    return current.min() < 0 and current.max() > 0


def _scan_range(df, potential_window=0.005) -> tuple[float, float]:
    current, potential = df["current"].values, df["potential"].values

    # find rough open circuit potential -- find zero crossing of current trace
    # if the LPR fit is any good, then the intercept should give
    # a more precise estimate of the open circuit potential
    zcross = np.argmin(np.abs(current))
    ocp = potential[zcross]

    # select a window around OCP to fit
    lb, ub = ocp - potential_window, ocp + potential_window
    return lb, ub


def valid_scan_range(df: EchemData, potential_window: float = 0.005) -> bool:
    """ check that an LPR scan has sufficient coverage around the open circuit potential """
    current, potential = df["current"], df["potential"]
    lb, ub = _scan_range(df, potential_window=potential_window)

    return potential.min() <= lb and potential.max() >= ub


def best_lpr_fit(df: EchemData, potential_window, r2_thresh=0.95):

    # straightforward linear fit
    slope, intercept, r2 = polarization_resistance(df, potential_window)

    current, potential, time = (
        df["current"].values,
        df["potential"].values,
        df["elapsed_time"].values,
    )
    fit_current = current
    if r2 < r2_thresh:
        ps_list = [0, 0.33, 0.66]
        best_chisq = np.inf
        result = None
        for ps in ps_list:
            chisq, dc_current = sin_fit(time, current, phase_shift=ps * np.pi * 2)
            if chisq < best_chisq:
                result = dc_current
                best_chisq = chisq
        dc_current = result

        corrected = pd.DataFrame({"current": dc_current, "potential": potential})
        slope2, intercept2, r22 = polarization_resistance(corrected, potential_window)
        if r22 > r2:
            slope = slope2
            intercept = intercept2
            r2 = r22
            fit_current = dc_current
    return slope, intercept, r2, fit_current


def sinfun(x, amp, afreq, bfreq, phaseshift):
    return amp * np.sin(x * (afreq * x + bfreq) + phaseshift)


def sin_fit(time, current, phase_shift=0):

    # aliases to make lmfit code more idiomatic...
    x, y = time, current

    mod = lmfit.models.PolynomialModel(5, prefix="bkgd_")
    pars = mod.guess(y, x=x)
    sinmodel = lmfit.Model(sinfun, prefix="sin_")
    mod += sinmodel
    # sinpars=sinmodel.make_params(amp=(np.max(y)-np.min(y))/4,freq=1/20*2*np.pi,phaseshift=0)
    sinpars = sinmodel.make_params(
        amp=(np.max(y) - np.min(y) - pars["bkgd_c0"] * np.max(x)) / 2,
        bfreq=1 / 20 * 2 * np.pi,
        phaseshift=phase_shift,
        afreq=0,
    )
    sinpars["sin_phaseshift"].min = 0
    sinpars["sin_phaseshift"].max = 2 * np.pi
    sinpars["sin_bfreq"].min = 0
    # sinpars['sin_bfreq'].max=1
    sinpars["sin_amp"].min = 0
    sinpars["sin_amp"].max = (np.max(y) - np.min(y)) / 2
    pars += sinpars
    out = mod.fit(y, pars, x=x, method="nelder")
    comps = out.eval_components(x=x)
    y_real = y - comps["sin_"]
    dc_current = comps["bkgd_"]
    chiaq = out.chisqr
    return chiaq, dc_current


def polarization_resistance(
    df: EchemData, potential_window: float = 0.005
) -> tuple[float, float, float]:
    """extract polarization resistance: fit a linear model relating measured current to potential

    Arguments:
        df: polarization resistance scan data
        potential_window: symmetric potential range around open circuit potential to fit polarization resistance model

    """

    current, potential = df["current"].values, df["potential"].values

    lb, ub = _scan_range(df, potential_window=potential_window)
    fit_p = (potential >= lb) & (potential <= ub)

    # quick linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        current[fit_p], potential[fit_p]
    )

    r2 = r_value ** 2

    return slope, intercept, r2


class LPRData(EchemData):
    @property
    def _constructor(self):
        return LPRData

    @property
    def name(self):
        return "LPR"

    def check_quality(df, r2_thresh=0.95, w=5):
        """ log results of quality checks and return a status code for control flow in the caller """

        status = Status.OK

        if not current_crosses_zero(df):
            logger.warning("LPR current does not cross zero!")
            status = max(status, Status.WARN)

        if not valid_scan_range(df, potential_window=w * 1e-3):
            logger.warning(f"scan range does not span +/- {w} mV")
            status = max(status, Status.WARN)

        slope, intercept, r2 = polarization_resistance(df)

        if r2 < r2_thresh:
            logger.warning("R^2 threshold not met")
            status = max(status, Status.WARN)

        logger.info(f"LPR slope: {slope} (R2={r2}), OCP: {intercept}")
        return status

    def fit(self):
        """ fit a polarization resistance model (linear model in +/- 5mV of OCP) """
        slope, intercept, r2, fit_current = best_lpr_fit(self, 0.005)

        self.polarization_resistance = slope
        self.open_circuit_potential = intercept
        self.r_value = r2
        self.fit_current = fit_current

        return slope, intercept, r2

    def evaluate_model(self, x):
        """ evaluate the fitted linear model """
        return self.open_circuit_potential + self.polarization_resistance * x

    def plot(self, fit=False):
        """LPR plot: plot current vs potential

        Optional: plot a regression line computing the polarization resistance
        """
        # # super().plot('current', 'potential')
        plt.plot(self["current"], self["potential"], ".")
        plt.axvline(0, color="k", alpha=0.5, linewidth=0.5)
        plt.xlabel("current (A)")
        plt.ylabel("potential (V)")

        if fit:
            self.fit()

            ylim = plt.ylim()
            x = np.linspace(self.current.min(), self.current.max(), 100)
            plt.plot(x, self.evaluate_model(x), linestyle="--", color="k", alpha=0.5)
            plt.plot(self.fit_current, self["potential"])
            plt.ylim(ylim)

        plt.tight_layout()
