import numpy as np
import sqlite3
from scipy.signal import lfilter
from filters import a_weighting_filter, c_weighting_filter  # ты можешь использовать свои
from time_utils import format_db_timestamp

def apply_weighting(signal, filter_coeffs):
    """Фильтрация сигнала с A/C коррекцией"""
    b, a = filter_coeffs
    return lfilter(b, a, signal)

def log_weighted_to_db(kind, spl, leq, lmax):
    conn = sqlite3.connect("sound_log.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO weighted_measurements (timestamp, weight_type, spl, leq, lmax)
        VALUES (?, ?, ?, ?, ?)""",
        (format_db_timestamp(), kind, spl, leq, lmax))
    conn.commit()
    conn.close()
