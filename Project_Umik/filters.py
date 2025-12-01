import numpy as np
from scipy.signal import bilinear, lfilter

def a_weighting_filter(fs=48000):
    # IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996
    f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194.0
    A1000 = 1.9997

    NUMs = [(2 * np.pi * f4)**2 * (10**(A1000 / 20.0)), 0, 0, 0, 0]
    DENs = np.polymul([1, 4 * np.pi * f4],
                      [1, 4 * np.pi * f1])
    DENs = np.polymul(np.polymul(DENs, [1, 2 * np.pi * f3]),
                                   [1, 2 * np.pi * f2])

    b, a = bilinear(NUMs, DENs, fs)
    return b, a


def c_weighting_filter(fs=48000):
    f1, f4 = 20.6, 12194.0
    C1000 = 0.0619

    NUMs = [(2 * np.pi * f4)**2 * (10**(C1000 / 20.0)), 0, 0]
    DENs = np.polymul([1, 4 * np.pi * f4],
                      [1, 4 * np.pi * f1])

    b, a = bilinear(NUMs, DENs, fs)
    return b, a
