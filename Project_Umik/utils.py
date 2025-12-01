import numpy as np

def db_to_linear(db):
    """
    Перевод dB в линейную шкалу.
    """
    return 10 ** (db / 20)

def linear_to_db(val, reference=1.0):
    """
    Перевод линейного значения в dB относительно reference.
    """
    if val <= 0:
        return -np.inf
    return 20 * np.log10(val / reference)

def normalize_signal(signal):
    """
    Нормализует сигнал к диапазону [-1, 1].
    """
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val != 0 else signal

def apply_ema(signal, alpha=0.125):
    """
    Применяет экспоненциальное скользящее среднее (EMA), используется для Fast/Slow.

    Args:
        signal (np.ndarray): Сигнал в Паскалях.
        alpha (float): Коэффициент EMA. Fast = ~0.125, Slow = ~0.03125

    Returns:
        np.ndarray: Отфильтрованный сигнал.
    """
    ema = np.zeros_like(signal)
    ema[0] = signal[0]
    for i in range(1, len(signal)):
        ema[i] = alpha * signal[i] + (1 - alpha) * ema[i - 1]
    return ema
