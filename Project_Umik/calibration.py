import numpy as np

def apply_calibration(signal_volts, sensitivity_v_per_pa):
    """
    Преобразует сигнал из Вольт в Паскали, используя чувствительность микрофона.

    Args:
        signal_volts (np.ndarray): Сырой сигнал (в В).
        sensitivity_v_per_pa (float): Чувствительность микрофона (В/Па), например 0.0045 В/Па.

    Returns:
        np.ndarray: Сигнал в Паскалях (Па).
    """
    if sensitivity_v_per_pa <= 0:
        raise ValueError("Неверная чувствительность микрофона")

    signal_pa = signal_volts / sensitivity_v_per_pa
    return signal_pa
