# logger_sqlite.py

import sqlite3
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

    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            max_leq REAL,
            threshold REAL,
            audio_file TEXT
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


def insert_event_start(start_time: str, threshold: float, audio_file: str | None = None) -> int:

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO events (start_time, end_time, max_leq, threshold, audio_file)
        VALUES (?, NULL, NULL, ?, ?)
    """, (start_time, threshold, audio_file))
    event_id = c.lastrowid
    conn.commit()
    conn.close()
    return event_id


def update_event_end(event_id: int, end_time: str, max_leq: float):

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        UPDATE events
        SET end_time = ?, max_leq = ?
        WHERE id = ?
    """, (end_time, max_leq, event_id))
    conn.commit()
    conn.close()
