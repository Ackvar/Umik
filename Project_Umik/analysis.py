import numpy as np

def compute_spl(signal_pa, reference_pressure=20e-6):
    """
    Вычисляет уровень звукового давления (SPL) в дБ.

    Args:
        signal_pa (np.ndarray): Сигнал в Паскалях.
        reference_pressure (float): Опорное давление, по умолчанию 20 мкПа.

    Returns:
        float: SPL в дБ (A-weighted, если применено A-взвешивание).
    """
    rms = np.sqrt(np.mean(signal_pa ** 2))
    if rms == 0:
        return -np.inf
    return 20 * np.log10(rms / reference_pressure)

def compute_leq(signal_pa, reference_pressure=20e-6):
    """
    Вычисляет Leq (эквивалентный уровень) за весь сигнал.

    Args:
        signal_pa (np.ndarray): Сигнал в Паскалях.
        reference_pressure (float): Опорное давление.

    Returns:
        float: Leq в дБ.
    """
    return compute_spl(signal_pa, reference_pressure)

def compute_lmax(signal_pa, reference_pressure=20e-6):
    """
    Вычисляет максимальный уровень звукового давления (Lmax).

    Args:
        signal_pa (np.ndarray): Сигнал в Паскалях.
        reference_pressure (float): Опорное давление.

    Returns:
        float: Lmax в дБ.
    """
    peak = np.max(np.abs(signal_pa))
    if peak == 0:
        return -np.inf
    return 20 * np.log10(peak / reference_pressure)
