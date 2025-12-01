import csv
import os

CSV_FILE = "sound_log.csv"

def init_csv():
    """
    Инициализирует CSV-файл с заголовками.
    """
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = ["Timestamp", "SPL", "Leq_1s", "Leq_60s", "Lmax"]
            # Добавим 1/1-октавные полосы в заголовки
            octave_freqs = ['31.5', '63', '125', '250', '500', '1000', '2000', '4000', '8000', '16000']
            headers.extend([f"{freq} Hz" for freq in octave_freqs])
            writer.writerow(headers)

def log_to_csv(timestamp, spl, leq_1s, leq_60s, lmax, bands):
    """
    Записывает строку данных в CSV-файл.
    """
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        row = [timestamp, f"{spl:.2f}", f"{leq_1s:.2f}", f"{leq_60s:.2f}", f"{lmax:.2f}"]

        # Дополнительно записываем значения по частотам
        octave_freqs = ['31.5', '63', '125', '250', '500', '1000', '2000', '4000', '8000', '16000']
        for freq in octave_freqs:
            value = bands.get(f"{freq} Hz", 0.0)
            row.append(f"{value:.2f}")
        writer.writerow(row)
