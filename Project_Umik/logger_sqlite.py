# logger_sqlite.py
import os
import sqlite3
import threading
from datetime import datetime

DB_NAME = "sound_log.db"
BUSY_TIMEOUT_MS = 30000
OCTAVE_COLUMNS = [f"{cf:.1f}_Hz" for cf in [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]]
_RECOVERY_LOCK = threading.RLock()


def _is_malformed_error(exc: Exception) -> bool:
    return "malformed" in str(exc).lower()


def _quarantine_corrupt_db(reason: Exception) -> None:
    with _RECOVERY_LOCK:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        moved = []
        for suffix in ("", "-wal", "-shm"):
            src = f"{DB_NAME}{suffix}"
            if not os.path.exists(src):
                continue
            dst = f"{DB_NAME}.corrupt.{stamp}{suffix}"
            os.replace(src, dst)
            moved.append(dst)
        print(f"[DB] corrupt database quarantined ({reason}); moved={moved}")


def _connect():
    conn = sqlite3.connect(DB_NAME, timeout=30)
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    return conn


def _init_schema(conn):
    c = conn.cursor()

    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")

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


def _quick_check_ok() -> bool:
    if not os.path.exists(DB_NAME):
        return False
    try:
        conn = _connect()
        try:
            row = conn.execute("PRAGMA quick_check;").fetchone()
            return bool(row and row[0] == "ok")
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False


def init_db():
    try:
        conn = _connect()
        try:
            _init_schema(conn)
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        if not _is_malformed_error(e):
            raise
        recover_corrupt_db(e)


def recover_corrupt_db(reason: Exception) -> None:
    with _RECOVERY_LOCK:
        if _quick_check_ok():
            return
        _quarantine_corrupt_db(reason)
        conn = _connect()
        try:
            _init_schema(conn)
        finally:
            conn.close()


def log_to_db(timestamp, spl, leq_1s, leq_60s, lmax, bands: dict):
    conn = _connect()
    try:
        values = [timestamp, spl, leq_1s, leq_60s, lmax]
        for cf in [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]:
            key = f"{cf:.1f} Hz"
            values.append(float(bands.get(key, 0.0)))

        placeholders = ",".join("?" for _ in values)
        conn.execute(f"INSERT INTO measurements VALUES ({placeholders})", values)
        conn.commit()
    finally:
        conn.close()


def insert_event_start(start_ts: str, threshold: float, audio_path: str) -> int:
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO noise_events (start_ts, threshold, audio_path) VALUES (?, ?, ?)",
            (start_ts, threshold, audio_path),
        )
        event_id = cur.lastrowid
        conn.commit()
        return event_id
    finally:
        conn.close()


def update_event_end(event_id: int, end_ts: str, max_leq: float) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE noise_events SET end_ts = ?, max_leq = ? WHERE id = ?",
            (end_ts, max_leq, event_id),
        )
        conn.commit()
    finally:
        conn.close()
