import argparse
from datetime import datetime
from scipy.io import wavfile
from calibration import apply_calibration
from weighting import apply_weighting
from analysis import compute_spl, compute_leq
import sqlite3
import numpy as np

def analyze_and_log(filename, weight_type="A", calibration_db=-26.0):
    fs, raw = wavfile.read(filename)
    signal = raw.astype(np.float32) / 32768.0
    calibrated = apply_calibration(signal, calibration_db)
    weighted = apply_weighting(calibrated, weight_type)

    spl = compute_spl(weighted)
    leq = compute_leq(weighted)
    lmax = np.max(weighted)

    # Log to database
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("sound_log.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS weighted_measurements (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        filename TEXT,
        weight_type TEXT,
        spl REAL,
        leq REAL,
        lmax REAL
    )""")
    c.execute("INSERT INTO weighted_measurements VALUES (NULL, ?, ?, ?, ?, ?, ?)",
              (ts, filename, weight_type, spl, leq, lmax))
    conn.commit()
    conn.close()

    print(f"✔️ {weight_type}-взвешенный SPL: {spl:.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="WAV-файл для анализа")
    parser.add_argument("--weight", choices=["A", "C", "Z"], default="A", help="Тип частотной коррекции")
    parser.add_argument("--cal", type=float, default=-26.0, help="Калибровка в dB (пример: -26.0)")
    args = parser.parse_args()
    analyze_and_log(args.filename, args.weight, args.cal)
