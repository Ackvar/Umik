# logger_sqlite.py

import sqlite3

DB_NAME = "sound_log.db"

OCTAVE_COLUMNS = [f"{cf:.1f}_Hz" for cf in
                  [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]]


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Основная таблица измерений
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

    # Таблица событий шума (для аудио)
    c.execute("""
        CREATE TABLE IF NOT EXISTS noise_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_ts   TEXT,
            end_ts     TEXT,
            threshold  REAL,
            max_leq    REAL,
            audio_path TEXT
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
    """Создаёт запись события и возвращает его id."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO noise_events (start_ts, threshold, audio_path) VALUES (?, ?, ?)",
        (start_ts, threshold, audio_path),
    )
    event_id = c.lastrowid
    conn.commit()
    conn.close()
    return event_id


def update_event_end(event_id: int, end_ts: str, max_leq: float) -> None:
    """Обновляет запись события концом и максимумом."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "UPDATE noise_events SET end_ts = ?, max_leq = ? WHERE id = ?",
        (end_ts, max_leq, event_id),
    )
    conn.commit()
    conn.close()
