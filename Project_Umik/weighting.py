import numpy as np
from scipy.signal import bilinear, lfilter

def design_a_weighting(fs):
    """
    Создаёт фильтр A-weighting согласно IEC/CD 1672.

    Args:
        fs (int): Частота дискретизации (обычно 48000 Гц).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Коэффициенты фильтра (b, a)
    """
    # Частоты, определённые в стандарте
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    nums = [(2 * np.pi * f4)**2 * (10**(A1000 / 20)),
            0, 0, 0, 0]
    dens = np.polymul(
        [1, 4 * np.pi * f4, (2 * np.pi * f4)**2],
        [1, 4 * np.pi * f1, (2 * np.pi * f1)**2]
    )
    dens = np.polymul(
        np.polymul(dens, [1, 2 * np.pi * f3]),
        [1, 2 * np.pi * f2]
    )

    # Преобразуем аналоговый фильтр в цифровой (bilinear transform)
    b, a = bilinear(nums, dens, fs)
    return b, a

def apply_a_weighting(signal, fs):
    """
    Применяет A-взвешивание к сигналу.

    Args:
        signal (np.ndarray): Сигнал в Паскалях.
        fs (int): Частота дискретизации.

    Returns:
        np.ndarray: Взвешенный сигнал.
    """
    b, a = design_a_weighting(fs)
    return lfilter(b, a, signal)
