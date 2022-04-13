import lmfit
import numpy as np
from scipy import signal
from scipy.spatial import distance
from sklearn import linear_model
import matplotlib.pyplot as plt

from sizedet.utils import *
from sizedet.constants import lambda_cu_Ka

def model_peak(x, y, peak, h, w, dw):

    # set up an lmfit model
    bg_mod = lmfit.models.LinearModel(prefix='bg_')
    # bg_mod = lmfit.models.PolynomialModel(2, prefix='bg_')
    pars = bg_mod.guess(y, x=x)

    prefix = f'peak_'
    gauss = lmfit.models.PseudoVoigtModel(prefix=prefix)
    pars.update(gauss.make_params())

    pars[f'{prefix}center'].set(value=peak)
    pars[f'{prefix}sigma'].set(value=w*dw/2, max=w*dw*3/2)
    pars[f'{prefix}amplitude'].set(value=h, min=1)
    model = bg_mod + gauss

    return model, pars

def model_blockwise(pattern, prominence=20, peak_whitelist=None, window=5):
    """ if peak whitelist is not none, filter peak detections to closest matches in whitelist """

    # convert to two theta copper K \alpha
    tt = 2*d2theta(10*pattern.Q.values)
    i = pattern.I

    # use peak detection to get rough guesses for the global model
    guesses, info = signal.find_peaks(i, prominence=prominence, height=1, width=1, rel_height=0.5)

    tt_peaks = tt[guesses]
    delta_w = np.diff(tt)[guesses]

    if peak_whitelist is not None:
        peak_idx = distance.cdist(2*d2theta(10*peak_whitelist)[:,None], tt_peaks[:,None]).argmin(axis=1)
        guesses = guesses[peak_idx]
        tt_peaks = tt_peaks[peak_idx]
        delta_w = delta_w[peak_idx]
        info['peak_heights'] = info['peak_heights'][peak_idx]
        info['widths'] = info['widths'][peak_idx]

    models = []
    results = []
    x = []
    for idx, (peak, h, w, dw) in enumerate(zip(tt_peaks, info['peak_heights'], info['widths'], delta_w)):
        sel = (tt > peak-window) & (tt < peak+window)
        m, pars = model_peak(tt[sel], i[sel], peak, h, w, dw)
        models.append(m)
        res = m.fit(i[sel], pars, x=tt[sel])
        results.append(res)
        x.append(tt[sel])

    return models, results, x

def wh(models, K=0.9, robust=False, plot=True, prefix='peak_'):

    # get the list of peak locations and widths
    n_peaks = len(models)
    loc = np.array([res.params[f'{prefix}center'] for res in models])
    fwhm = np.array([res.params[f'{prefix}fwhm'] for res in models])

    if robust:
        m = linear_model.HuberRegressor()
    else:
        m = linear_model.LinearRegression()

    xx = np.sin(np.radians(loc/2))
    yy = np.radians(fwhm)*np.cos(np.radians(loc/2))
    m.fit(xx[:,None], yy)

    if plot:
        plt.scatter(xx, yy)
        plt.xlabel(r'$sin(\theta)$')
        plt.ylabel(r'fwhm * $cos(\theta)$');

        _x = np.linspace(0, 1, 100)
        plt.plot(_x, m.predict(_x[:,None]), color='k', linestyle='--');

    crystallite_size = K*lambda_cu_Ka / m.intercept_
    strain = m.coef_[0]/4

    return crystallite_size, strain

def model_peak_complex(pattern, prominence=20, primary_peak=None):
    """ fit primary_peak """

    # convert to two theta copper K \alpha
    tt = 2*d2theta(10*pattern.Q.values)
    i = pattern.I

    # use peak detection to get rough guesses for the global model
    guesses, info = signal.find_peaks(i, prominence=prominence, height=1, width=1, rel_height=0.5)

    tt_peaks = tt[guesses]
    delta_w = np.diff(tt)[guesses]

    if primary_peak is not None:
        primary_peak_idx = np.abs(2*d2theta(10*primary_peak) - tt_peaks).argmin()

    # set up an lmfit model
    # bg_mod = lmfit.models.LinearModel(prefix='bg_')
    bg_mod = lmfit.models.PolynomialModel(2, prefix='bg_')
    pars = bg_mod.guess(i, x=tt)

    model = bg_mod

    for idx, (peak, h, w, dw) in enumerate(zip(tt_peaks, info['peak_heights'], info['widths'], delta_w)):
        if idx == primary_peak_idx:
            prefix = 'primary_peak_'
        else:
            prefix = f'g{idx}_'
        gauss = lmfit.models.PseudoVoigtModel(prefix=prefix)
        pars.update(gauss.make_params())

        pars[f'{prefix}center'].set(value=peak)
        pars[f'{prefix}sigma'].set(value=w*dw/2, max=w*dw*3/2)
        pars[f'{prefix}amplitude'].set(value=h, min=1)
        model = model + gauss

    return model, pars

def model_blockwise_complex(pattern, prominence=20, peak_whitelist=None, window=5):
    """ if peak whitelist is not none, filter peak detections to closest matches in whitelist """

    # convert to two theta copper K \alpha
    tt = 2*d2theta(10*pattern.Q.values)
    i = pattern.I

    # use peak detection to get rough guesses for the global model
    guesses, info = signal.find_peaks(i, prominence=prominence, height=1, width=1, rel_height=0.5)

    tt_peaks = tt[guesses]
    delta_w = np.diff(tt)[guesses]

    if peak_whitelist is not None:
        peak_idx = distance.cdist(2*d2theta(10*peak_whitelist)[:,None], tt_peaks[:,None]).argmin(axis=1)
        guesses = guesses[peak_idx]
        tt_peaks = tt_peaks[peak_idx]
        delta_w = delta_w[peak_idx]
        info['peak_heights'] = info['peak_heights'][peak_idx]
        info['widths'] = info['widths'][peak_idx]

    models = []
    results = []
    x = []
    for idx, (peak, h, w, dw) in enumerate(zip(tt_peaks, info['peak_heights'], info['widths'], delta_w)):
        sel = (tt > peak-window) & (tt < peak+window)
        m, pars = model_peak_complex(pattern.iloc[sel], primary_peak=peak_whitelist[idx], prominence=prominence)
        models.append(m)
        res = m.fit(i[sel], pars, x=tt[sel])
        results.append(res)
        x.append(tt[sel])

    return models, results, x

def model_xrd(pattern, prominence=20, peak_whitelist=None):
    """ if peak whitelist is not none, filter peak detections to closest matches in whitelist """

    # convert to two theta copper K \alpha
    tt = 2*d2theta(10*pattern.Q.values)
    i = pattern.I

    # use peak detection to get rough guesses for the global model
    guesses, info = signal.find_peaks(i, prominence=prominence, height=1, width=1, rel_height=0.5)

    tt_peaks = tt[guesses]
    delta_w = np.diff(tt)[guesses]

    if peak_whitelist is not None:
        peak_idx = distance.cdist(2*d2theta(10*peak_whitelist)[:,None], tt_peaks[:,None]).argmin(axis=1)
        guesses = guesses[peak_idx]
        tt_peaks = tt_peaks[peak_idx]
        delta_w = delta_w[peak_idx]
        info['peak_heights'] = info['peak_heights'][peak_idx]
        info['widths'] = info['widths'][peak_idx]

    # set up an lmfit model
    # bg_mod = lmfit.models.ConstantModel(prefix='bg_')
    bg_mod = lmfit.models.LinearModel(prefix='bg_')
    # bg_mod = lmfit.models.PolynomialModel(3, prefix='bg_')
    pars = bg_mod.guess(i, x=tt)

    model = bg_mod

    for idx, (peak, h, w, dw) in enumerate(zip(tt_peaks, info['peak_heights'], info['widths'], delta_w)):
        prefix = f'g{idx}_'
        gauss = lmfit.models.PseudoVoigtModel(prefix=prefix)
        pars.update(gauss.make_params())

        pars[f'{prefix}center'].set(value=peak)
        pars[f'{prefix}sigma'].set(value=w*dw/2, max=w*dw*3/2)
        pars[f'{prefix}amplitude'].set(value=h, min=1)
        model = model + gauss

    return model, pars, tt, info

def williamson_hall(tt, model, res, K=0.9, robust=False, plot=True):

    # get the list of peak locations and widths
    n_peaks = len(model.components) - 1
    loc = np.array([res.params[f'g{idx}_center'] for idx in range(n_peaks)])
    fwhm = np.array([res.params[f'g{idx}_fwhm'] for idx in range(n_peaks)])
    # print(loc)
    # print(fwhm)

    if robust:
        m = linear_model.HuberRegressor()
    else:
        m = linear_model.LinearRegression()


    xx = np.sin(np.radians(loc/2))
    yy = np.radians(fwhm)*np.cos(np.radians(loc/2))
    m.fit(xx[:,None], yy)

    if plot:
        plt.scatter(xx, yy)
        plt.xlabel(r'$sin(\theta)$')
        plt.ylabel(r'fwhm * $cos(\theta)$');

        _x = np.linspace(0, 1, 100)
        plt.plot(_x, m.predict(_x[:,None]), color='k', linestyle='--');

    crystallite_size = K*lambda_cu_Ka / m.intercept_
    strain = m.coef_[0]/4

    return crystallite_size, strain
