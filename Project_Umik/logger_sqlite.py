# logger_sqlite.py
from __future__ import annotations
import sqlite3
from pathlib import Path

DB_NAME = "sound_log.db"

OCTAVE_COLUMNS = [f"{cf:.1f}_Hz" for cf in [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]]

def init_db():
    Path(DB_NAME).touch(exist_ok=True)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # таблица измерений (как было)
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

    # таблица событий превышения
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_ts TEXT,
            end_ts TEXT,
            threshold REAL,
            max_leq REAL,
            audio_path TEXT,
            measurement_id INTEGER
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
    c.execute(f"INSERT INTO measurements VALUES ({placeholders})", values)
    conn.commit()
    conn.close()


def insert_event_start(start_ts: str, threshold: float, audio_path: str) -> int:
    """
    Создаёт событие превышения шума, возвращает event_id.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO events (start_ts, threshold, audio_path) VALUES (?, ?, ?)",
        (start_ts, threshold, audio_path),
    )
    event_id = c.lastrowid
    conn.commit()
    conn.close()
    return event_id


def update_event_end(event_id: int, end_ts: str, max_leq: float, measurement_id: int | None):
    """
    Завершает событие: пишет конец, max_leq и id измерения на сервере (если есть).
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "UPDATE events SET end_ts = ?, max_leq = ?, measurement_id = ? WHERE id = ?",
        (end_ts, max_leq, measurement_id, event_id),
    )
    conn.commit()
    conn.close()
