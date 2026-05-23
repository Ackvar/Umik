# app.py
from __future__ import annotations
from flask import Flask, render_template, jsonify, request
import sqlite3
import os
import time
import json
import threading
import requests
from datetime import datetime, timedelta
from urllib.parse import quote
from state import get_fft
from logger_sqlite import BUSY_TIMEOUT_MS, DB_NAME, recover_corrupt_db
from time_utils import APP_TIMEZONE, app_now, format_db_timestamp, to_utc_iso

# ========= Настройки внешней отправки =========
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL")
EXTERNAL_API_TOKEN = os.getenv("EXTERNAL_API_TOKEN")
SEND_INTERVAL_SEC = float(os.getenv("SEND_INTERVAL_SEC", "1.0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))

app = Flask(__name__)
_http = requests.Session()
_last_db_error_log = 0.0


def get_rpi_serial():
    serial = "UNKNOWN"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.strip().startswith("Serial"):
                    serial = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    return serial


# Настройки периодического capture-отчёта
REPORT_API_URL = os.getenv(
    "REPORT_API_URL",
    "https://shum.i20h.ru/api/v1/measurements/capture/"
)
REPORT_INTERVAL_SEC = int(os.getenv("REPORT_INTERVAL_SEC", "120"))  # 2 минуты
DEVICE_ID = os.getenv("DEVICE_ID", get_rpi_serial())  # device_serial

try:
    with open("umik_config.json", "r", encoding="utf-8") as f:
        _APP_CONFIG = json.load(f)
except Exception:
    _APP_CONFIG = {}

EVENT_THRESHOLD_DB = float(os.getenv("EVENT_THRESHOLD_DB", _APP_CONFIG.get("event_threshold_db", 45.0)))


# ========= Helpers =========

def db_rows(query: str, args: tuple = ()) -> list[sqlite3.Row]:
    global _last_db_error_log
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
        cur = conn.cursor()
        cur.execute(query, args)
        return cur.fetchall()
    except sqlite3.DatabaseError as e:
        now = time.time()
        if now - _last_db_error_log >= 30:
            print(f"[DB] read failed: {e}")
            _last_db_error_log = now
        if "malformed" in str(e).lower():
            if conn is not None:
                conn.close()
                conn = None
            try:
                recover_corrupt_db(e)
            except Exception as recover_error:
                print(f"[DB] recovery failed: {recover_error}")
        return []
    finally:
        if conn is not None:
            conn.close()


def get_last_measurements(limit: int = 20):
    rows = db_rows("SELECT * FROM measurements ORDER BY timestamp DESC LIMIT ?", (limit,))
    if not rows:
        return [], []
    columns = list(rows[0].keys())
    return columns, [tuple(r) for r in rows]


def _window_to_db_bounds(
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> tuple[datetime, datetime, str, str]:
    end_dt = window_end or app_now()
    start_dt = window_start or (end_dt - timedelta(seconds=REPORT_INTERVAL_SEC))
    return start_dt, end_dt, format_db_timestamp(start_dt), format_db_timestamp(end_dt)


def get_10min_report_measurement(
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> dict | None:
    """
    Pick one report row for the finished report window:
    1) highest threshold exceedance in the window;
    2) if there was no exceedance, highest value below the threshold.
    """
    start_dt, end_dt, start_ts, end_ts = _window_to_db_bounds(window_start, window_end)

    rows = db_rows(
        """
        SELECT timestamp, leq_1s
        FROM measurements
        WHERE timestamp >= ?
          AND timestamp < ?
          AND leq_1s IS NOT NULL
          AND leq_1s >= ?
        ORDER BY leq_1s DESC, timestamp DESC
        LIMIT 1
        """,
        (start_ts, end_ts, EVENT_THRESHOLD_DB),
    )
    exceeded = True

    if not rows:
        rows = db_rows(
            """
            SELECT timestamp, leq_1s
            FROM measurements
            WHERE timestamp >= ?
              AND timestamp < ?
              AND leq_1s IS NOT NULL
              AND leq_1s < ?
            ORDER BY leq_1s DESC, timestamp DESC
            LIMIT 1
            """,
            (start_ts, end_ts, EVENT_THRESHOLD_DB),
        )
        exceeded = False

    if not rows:
        return None

    row = rows[0]
    return {
        "value": row["leq_1s"],
        "timestamp": row["timestamp"],
        "exceeded": exceeded,
        "threshold": EVENT_THRESHOLD_DB,
        "window_start": start_dt,
        "window_end": end_dt,
        "window_start_ts": start_ts,
        "window_end_ts": end_ts,
    }


def get_10min_max_level():
    """
    Возвращает (max_leq, ts_at_max) за последнее окно отчёта.
    """
    selected = get_10min_report_measurement()
    if not selected:
        return None, None
    return selected["value"], selected["timestamp"]


def _to_iso(ts_str: str) -> str:
    """Преобразует локальное время проекта в ISO8601 UTC."""
    return to_utc_iso(ts_str)


def _report_capture_url() -> str:
    base = REPORT_API_URL.rstrip("/")
    device_serial = str(DEVICE_ID).strip()
    encoded_serial = quote(device_serial, safe="")
    last_segment = base.rsplit("/", 1)[-1]
    if last_segment in {device_serial, encoded_serial}:
        return f"{base}/"
    return f"{base}/{encoded_serial}/"


# ========= Периодический capture-отчёт =========

def _extract_measurement_id(data):
    if isinstance(data, list) and data:
        return data[0].get("id")
    if isinstance(data, dict):
        return data.get("id") or data.get("measurement")
    return None


def _send_measurement_json(value: float, event_ts: str, *, prefix: str, timeout: float = 10) -> tuple[bool, int | None]:
    event_time_iso = _to_iso(event_ts)
    payload = [{
        "value": float(value),
        "event_time": event_time_iso,
    }]

    try:
        resp = _http.post(_report_capture_url(), json=payload, timeout=timeout)
        if not (200 <= resp.status_code < 300):
            print(f"{prefix} FAIL JSON {resp.status_code}: {resp.text}")
            return False, None

        try:
            data = resp.json()
        except Exception:
            print(f"{prefix} JSON OK, but response is not JSON: {resp.text}")
            return True, None

        measurement_id = _extract_measurement_id(data)
        if measurement_id is None:
            print(f"{prefix} JSON OK, measurement id was not returned: {data}")
            return True, None
        return True, int(measurement_id)
    except Exception as e:
        print(f"{prefix} ERROR JSON: {e}")
        return False, None


def send_audio_for_measurement(measurement_id: int, audio_path: str, *, prefix: str = "[EVENT]") -> bool:
    if not os.path.exists(audio_path):
        print(f"{prefix} Файл аудио не найден: {audio_path}")
        return False

    base = _report_capture_url().rstrip("/")
    audio_url = f"{base}/{measurement_id}/audio/"
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            resp = _http.post(audio_url, files=files, timeout=30)
        if 200 <= resp.status_code < 300:
            print(f"{prefix} AUDIO OK id={measurement_id} file={audio_path}")
            return True
        print(f"{prefix} AUDIO FAIL {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"{prefix} ERROR при отправке аудио: {e}")
    return False


def send_10min_report(
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    audio_lookup=None,
):
    """
    Формирует и отправляет JSON вида (список!):
    [
      {
        "value": <максимальный leq_1s>,
        "event_time": "ISO-время этого максимума"
      }
    ]
    """
    if not REPORT_API_URL:
        print("[REPORT] REPORT_API_URL не задан, отправка отключена")
        return

    selected = get_10min_report_measurement(window_start, window_end)
    if not selected:
        print("[REPORT] За окно измерений данных нет, JSON не отправляем")
        return

    value = float(selected["value"])
    event_ts = selected["timestamp"]
    audio_path = None
    if selected["exceeded"] and audio_lookup:
        try:
            audio_path = audio_lookup(selected)
        except Exception as e:
            print(f"[REPORT] AUDIO lookup error: {e}")

    json_ok, measurement_id = _send_measurement_json(value, event_ts, prefix="[REPORT]", timeout=10)
    if not json_ok:
        return

    state = "exceedance" if selected["exceeded"] else "below-threshold"
    print(
        f"[REPORT] OK {state} value={value:.2f} dB at {event_ts} "
        f"window={selected['window_start_ts']}..{selected['window_end_ts']}"
    )

    if audio_path:
        if measurement_id is None:
            print(f"[REPORT] AUDIO skipped: measurement id was not returned for {event_ts}")
            return
        send_audio_for_measurement(measurement_id, audio_path, prefix="[REPORT]")


def _next_report_boundary(now: datetime | None = None) -> datetime:
    now = now or app_now()
    interval = max(1, REPORT_INTERVAL_SEC)
    next_epoch = (int(now.timestamp() // interval) + 1) * interval
    return datetime.fromtimestamp(next_epoch, tz=APP_TIMEZONE)


def report_loop(on_window_close=None, audio_lookup=None):
    if not REPORT_API_URL:
        print("[REPORT] REPORT_API_URL не задан, репортер не запущен")
        return

    print(f"[REPORT] Старт репортера: интервал {REPORT_INTERVAL_SEC} сек, "
          f"URL={_report_capture_url()}")
    next_window_end = _next_report_boundary()
    while True:
        sleep_for = (next_window_end - app_now()).total_seconds()
        if sleep_for > 0:
            time.sleep(sleep_for)

        window_end = next_window_end
        window_start = window_end - timedelta(seconds=REPORT_INTERVAL_SEC)

        if on_window_close:
            try:
                on_window_close(window_start, window_end)
            except Exception as e:
                print(f"[REPORT] on_window_close error: {e}")

        threading.Thread(
            target=send_10min_report,
            kwargs={
                "window_start": window_start,
                "window_end": window_end,
                "audio_lookup": audio_lookup,
            },
            daemon=True,
        ).start()

        next_window_end = next_window_end + timedelta(seconds=REPORT_INTERVAL_SEC)
        now = app_now()
        while next_window_end <= now:
            next_window_end = next_window_end + timedelta(seconds=REPORT_INTERVAL_SEC)


def start_reporter(on_window_close=None, audio_lookup=None):
    t = threading.Thread(
        target=report_loop,
        kwargs={"on_window_close": on_window_close, "audio_lookup": audio_lookup},
        daemon=True,
    )
    t.start()


# ========= Отправка события + аудио =========

def send_event_with_audio(value: float, event_ts: str, audio_path: str):
    """
    1) Шлёт JSON в REPORT_API_URL.
    2) Забирает id из ответа.
    3) Шлёт файл audio на /capture/{id}/audio/.
    event_ts — строка 'YYYY-MM-DD HH:MM:SS' (локальное время).
    """
    if not REPORT_API_URL:
        print("[EVENT] REPORT_API_URL не задан, отправка события отключена")
        return

    json_ok, measurement_id = _send_measurement_json(float(value), event_ts, prefix="[EVENT]", timeout=10)
    if not json_ok or measurement_id is None:
        return
    print(f"[EVENT] JSON OK, measurement_id={measurement_id}")
    send_audio_for_measurement(measurement_id, audio_path, prefix="[EVENT]")


# ========= Страницы =========

@app.route("/")
def index():
    columns, rows = get_last_measurements()
    return render_template("table.html", columns=columns, rows=rows)


@app.route("/table")
def table_page():
    columns, rows = get_last_measurements()
    return render_template("table.html", columns=columns, rows=rows)


@app.route("/chart")
def chart_view():
    return render_template("chart.html")


@app.route("/octave")
def octave_chart():
    return render_template("octave_chart.html")


@app.route("/rta")
def rta_view():
    return render_template("rta.html")


@app.route("/filtr")
def filtr_view():
    return render_template("filtr.html")


@app.route("/filter")
def filter_alias():
    return render_template("filtr.html")


# ========= API =========

@app.route("/api/latest")
def latest_data():
    rows = db_rows("SELECT timestamp, spl, leq_1s, lmax FROM measurements ORDER BY timestamp DESC LIMIT 60")
    data = {
        "timestamps": [r[0] for r in reversed(rows)],
        "spl": [r[1] for r in reversed(rows)],
        "leq": [r[2] for r in reversed(rows)],
        "lmax": [r[3] for r in reversed(rows)],
    }
    return jsonify(data)


@app.route("/api/octave")
def get_latest_octaves():
    rows = db_rows("SELECT * FROM measurements ORDER BY timestamp DESC LIMIT 1")
    if rows:
        row = rows[0]
        target_freqs = [
            "31.5 Hz", "63.0 Hz", "125.0 Hz", "250.0 Hz",
            "500.0 Hz", "1000.0 Hz", "2000.0 Hz", "4000.0 Hz", "8000.0 Hz",
        ]
        octave_values = row[-9:]
        return jsonify(dict(zip(target_freqs, octave_values)))
    return jsonify({})


@app.route("/api/fft")
def get_fft_api():
    data = get_fft()
    if data:
        return jsonify(data)
    return jsonify({"freqs": [], "values": []})


@app.get("/api/health")
def api_health():
    try:
        r = db_rows("SELECT COUNT(*) AS cnt FROM measurements")
        cnt = r[0]["cnt"] if r else 0
        return jsonify({"ok": True, "count": cnt, "external": bool(EXTERNAL_API_URL)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/last")
def api_last():
    r = db_rows("SELECT * FROM measurements ORDER BY timestamp DESC LIMIT 1")
    return jsonify(dict(r[0])) if r else jsonify({})


@app.get("/api/history")
def api_history():
    limit = int(request.args.get("limit", 200))
    rows = db_rows("SELECT * FROM measurements ORDER BY timestamp DESC LIMIT ?", (limit,))
    return jsonify([dict(r) for r in rows][::-1])


@app.get("/api/metrics")
def api_metrics():
    r = db_rows("SELECT * FROM measurements ORDER BY timestamp DESC LIMIT 1")
    if not r:
        return jsonify({})
    row = dict(r[0])
    keys = list(row.keys())
    oct_map = {
        "31.5": row.get(keys[-9], None),
        "63": row.get(keys[-8], None),
        "125": row.get(keys[-7], None),
        "250": row.get(keys[-6], None),
        "500": row.get(keys[-5], None),
        "1000": row.get(keys[-4], None),
        "2000": row.get(keys[-3], None),
        "4000": row.get(keys[-2], None),
        "8000": row.get(keys[-1], None),
    }
    payload = {
        "timestamp": row.get("timestamp"),
        "spl": row.get("spl"),
        "leq_1s": row.get("leq_1s"),
        "leq_60s": row.get("leq_60s") if "leq_60s" in row else None,
        "lmax": row.get("lmax"),
        "weighting": row.get("weighting") if "weighting" in row else None,
        "time_weighting": row.get("time_weighting") if "time_weighting" in row else None,
        "octaves": oct_map,
    }
    return jsonify(payload)


# ========= Старый "батчевый" отправщик (можно не использовать) =========

def _sender_loop():
    if not EXTERNAL_API_URL:
        return
    headers = {"Content-Type": "application/json"}
    if EXTERNAL_API_TOKEN:
        headers["Authorization"] = f"Bearer {EXTERNAL_API_TOKEN}"

    last_ts = None
    r = db_rows("SELECT timestamp FROM measurements ORDER BY timestamp DESC LIMIT 1")
    if r:
        last_ts = r[0]["timestamp"]

    while True:
        try:
            if last_ts is None:
                q = "SELECT * FROM measurements ORDER BY timestamp ASC LIMIT ?"
                args = (BATCH_SIZE,)
            else:
                q = "SELECT * FROM measurements WHERE timestamp > ? ORDER BY timestamp ASC LIMIT ?"
                args = (last_ts, BATCH_SIZE)

            new_rows = db_rows(q, args)
            if not new_rows:
                time.sleep(SEND_INTERVAL_SEC)
                continue

            payload = [dict(x) for x in new_rows]

            ok = False
            backoff = 1.0
            for _ in range(3):
                try:
                    resp = _http.post(EXTERNAL_API_URL, data=json.dumps(payload),
                                      headers=headers, timeout=10)
                    if 200 <= resp.status_code < 300:
                        ok = True
                        break
                except requests.RequestException:
                    pass
                time.sleep(backoff)
                backoff *= 2

            if ok:
                last_ts = new_rows[-1]["timestamp"]
            else:
                time.sleep(max(2.0, SEND_INTERVAL_SEC))
        except Exception:
            time.sleep(max(2.0, SEND_INTERVAL_SEC))


def start_sender():
    if EXTERNAL_API_URL:
        t = threading.Thread(target=_sender_loop, daemon=True)
        t.start()

# ========= Noise RAW (по PDF): отправка 1 раз в секунду =========

NOISE_RAW_API_URL = os.getenv("NOISE_RAW_API_URL", "https://int.kik.mos.ru/noise_raw_data")
NOISE_RAW_INTERVAL_SEC = float(os.getenv("NOISE_RAW_INTERVAL_SEC", "1.0"))

# координаты оборудования (если нет GPS — задай руками env-переменными)
LATITUDE_EQUIP = float(os.getenv("LATITUDE_EQUIP", "0"))
LONGITUDE_EQUIP = float(os.getenv("LONGITUDE_EQUIP", "0"))
ALTITUDE_EQUIP = float(os.getenv("ALTITUDE_EQUIP", "0"))

# UIN: минимум 1 значение. Можно хранить через запятую: "12345,67890"
UIN_LIST = [x.strip() for x in os.getenv("UIN_LIST", "0000000000").split(",") if x.strip()]

def _get_last_row():
    r = db_rows("SELECT * FROM measurements ORDER BY timestamp DESC LIMIT 1")
    return dict(r[0]) if r else None

def _build_noise_raw_payload(last_row: dict):
    """
    Формат из PDF: time_stamp, serial_number, las, dt, timestamp, message_type,
    latitude_equip, longitude_equip, altitude_equip, uin[]
    """
    now_epoch = int(time.time())
    dt_str = last_row.get("timestamp")  # строка времени из БД (как ты пишешь)
    # las — берём leq_1s как наиболее “1-сек” метрика, иначе spl
    las_val = last_row.get("leq_1s", None)
    if las_val is None:
        las_val = last_row.get("spl", 0)

    payload = {
        "time_stamp": now_epoch,               # как "метка времени" (epoch)
        "serial_number": DEVICE_ID,            # серийник RPi/устройства
        "las": float(las_val),                 # текущий уровень
        "dt": dt_str,                          # время измерения строкой
        "timestamp": now_epoch,                # epoch (часто требуют отдельно)
        "message_type": "noise_message",
        "latitude_equip": LATITUDE_EQUIP,
        "longitude_equip": LONGITUDE_EQUIP,
        "altitude_equip": ALTITUDE_EQUIP,
        "uin": UIN_LIST,                       # минимум одно значение
    }
    return payload

def _noise_raw_loop():
    if not NOISE_RAW_API_URL:
        print("[RAW] NOISE_RAW_API_URL не задан — RAW-отправка выключена")
        return

    headers = {"Content-Type": "application/json"}
    last_sent_ts = None  # чтобы не слать одно и то же много раз

    print(f"[RAW] Старт RAW-отправки: {NOISE_RAW_API_URL}, interval={NOISE_RAW_INTERVAL_SEC}s")

    while True:
        try:
            row = _get_last_row()
            if not row:
                time.sleep(NOISE_RAW_INTERVAL_SEC)
                continue

            # защита от дублей: сравниваем timestamp из БД
            ts = row.get("timestamp")
            if ts == last_sent_ts:
                time.sleep(NOISE_RAW_INTERVAL_SEC)
                continue

            payload = _build_noise_raw_payload(row)

            resp = _http.post(
                NOISE_RAW_API_URL,
                data=json.dumps(payload),
                headers=headers,
                timeout=10
            )

            if 200 <= resp.status_code < 300:
                last_sent_ts = ts
                # можно не спамить лог каждую секунду — но для отладки полезно:
                # print(f"[RAW] OK las={payload['las']:.1f} dt={payload['dt']}")
            else:
                print(f"[RAW] FAIL {resp.status_code}: {resp.text}")

        except Exception as e:
            print(f"[RAW] ERROR: {e}")

        time.sleep(NOISE_RAW_INTERVAL_SEC)

def start_noise_raw_sender():
    t = threading.Thread(target=_noise_raw_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    # start_sender()  # не нужен
    start_reporter()    
    start_noise_raw_sender()
    app.run(host="0.0.0.0", port=5000, debug=True)

