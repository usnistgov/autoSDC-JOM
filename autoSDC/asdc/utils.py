import os
import numpy as np


def make_circle(r, n=None):
    if n == None:
        dtheta = 0.01
    else:
        dtheta = 2 * np.pi / n

    t = np.arange(0, np.pi * 2.0, dtheta)
    t = t.reshape((len(t), 1))
    x = r * np.sin(t)
    y = r * np.cos(t)
    return np.hstack((x, y))
