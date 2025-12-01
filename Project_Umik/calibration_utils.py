import numpy as np
from scipy.interpolate import interp1d
from numpy.fft import rfft, irfft, rfftfreq


def load_calibration_curve(filepath):
    """
    Загружает калибровочную кривую из текстового файла.
    Ожидается две колонки: частота (Гц) и усиление (дБ).
    Возвращает массивы частот и коэффициентов усиления в линейной шкале.
    """
    freqs = []
    gains_db = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('"'):
                continue
            try:
                parts = line.split()
                if len(parts) == 2:
                    freq, gain = map(float, parts)
                    freqs.append(freq)
                    gains_db.append(gain)
            except ValueError:
                continue  # Пропускаем некорректные строки

    freqs = np.array(freqs)
    gains_db = np.array(gains_db)
    gains_linear = 10 ** (gains_db / 20)  # Преобразование из dB в линейную шкалу

    return freqs, gains_linear


def apply_frequency_calibration(signal, sample_rate, freqs, gains):
    """
    Применяет частотную калибровку к аудиосигналу.
    Сначала выполняется FFT, далее интерполяция усиления и обратный FFT.
    """
    N = len(signal)
    signal_fft = rfft(signal)
    freqs_fft = rfftfreq(N, d=1 / sample_rate)

    # Интерполяция коэффициентов усиления по частоте
    gain_interp = interp1d(freqs, gains, bounds_error=False, fill_value=(gains[0], gains[-1]))
    gain_curve = gain_interp(freqs_fft)

    # Применение калибровочной кривой к спектру
    calibrated_fft = signal_fft * gain_curve
    calibrated_signal = irfft(calibrated_fft, n=N)

    return calibrated_signal
