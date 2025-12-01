import numpy as np
from scipy.signal import butter, sosfilt

# Октавные центральные частоты (в Гц)
OCTAVE_BANDS = [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]


def bandpass_filter(data, fs, center_freq, bandwidth=1):
    factor = 2 ** (1 / (2 * bandwidth))
    low = center_freq / factor
    high = center_freq * factor
    sos = butter(N=4, Wn=[low, high], btype='bandpass', output='sos', fs=fs)
    return sosfilt(sos, data)


def octave_band_levels(signal, fs, reference_pressure=20e-6):
    results = {}
    for cf in OCTAVE_BANDS:
        filtered = bandpass_filter(signal, fs, cf, bandwidth=1)
        rms = np.sqrt(np.mean(filtered ** 2))
        if rms <= 0:
            level = -np.inf
        else:
            level = 20 * np.log10(rms / reference_pressure)

        level_float = float(round(level, 1))  # Приведение к обычному float
        results[f"{cf:.1f} Hz"] = level_float

        # DEBUG: распечатать для контроля
        print(f"[{cf:.1f} Hz] RMS={rms:.6f} → SPL={level_float:.2f} dB")

    return results
