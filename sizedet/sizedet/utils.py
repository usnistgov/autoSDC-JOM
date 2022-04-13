import numpy as np
from sizedet.constants import lambda_cu_Ka

"""
d = (4* np.pi * np.sin(angle)) / lambda_cu_Ka
d * lambda_cu_Ka = 4 * np.pi * np.sin(angle)
d * lambda_cu_Ka / (4 * np.pi) = np.sin(angle)
d * lambda_cu_Ka / (4 * np.pi) = np.sin(angle)
np.arcsin(d * lambda_cu_Ka / (4 * np.pi)) = angle
"""
def theta2d(angle):
    """ takes angle in radians """
    return (4* np.pi * np.sin(np.radians(angle))) / lambda_cu_Ka

def d2theta(d):
    angle = np.arcsin(d * lambda_cu_Ka / (4 * np.pi))
    return np.degrees(angle)
