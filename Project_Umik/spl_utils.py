import numpy as np

# Стандартное звуковое давление: 20 µPa
REFERENCE_PRESSURE = 20e-6

def compute_spl(signal, reference_pressure=REFERENCE_PRESSURE):
    """Рассчитывает SPL в dB"""
    rms = np.sqrt(np.mean(signal**2))
    if rms == 0:
        return -np.inf
    return 20 * np.log10(rms / reference_pressure)

def compute_leq(signal, reference_pressure=REFERENCE_PRESSURE):
    """Рассчитывает эквивалентный уровень звука"""
    return compute_spl(signal, reference_pressure)
