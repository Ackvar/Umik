# logger_sqlite.py

import sqlite3
import os
from datetime import datetime

DB_NAME = "sound_log.db"

OCTAVE_COLUMNS = [f"{cf:.1f}_Hz" for cf in [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]]

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS measurements (
            timestamp TEXT,
            spl REAL,
            leq_1s REAL,
            leq_60s REAL,
            lmax REAL,
            {', '.join([f'"{col}" REAL' for col in OCTAVE_COLUMNS])}
        );
    """)
    conn.commit()
    conn.close()

def log_to_db(timestamp, spl, leq_1s, leq_60s, lmax, bands):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    values = [timestamp, spl, leq_1s, leq_60s, lmax]
    for cf in [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]:
        key = f"{cf:.1f} Hz"
        values.append(float(bands.get(key, 0)))

    placeholders = ",".join("?" for _ in values)
    c.execute(f"""
        INSERT INTO measurements VALUES ({placeholders})
    """, values)
    conn.commit()
    conn.close()
