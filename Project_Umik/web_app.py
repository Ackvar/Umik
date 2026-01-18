# app.py
from __future__ import annotations
from flask import Flask, render_template, jsonify, request
import sqlite3
import os
import time
import json
import threading
import requests
import datetime
from state import get_fft

# ========= Настройки внешней отправки =========
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL")
EXTERNAL_API_TOKEN = os.getenv("EXTERNAL_API_TOKEN")
SEND_INTERVAL_SEC = float(os.getenv("SEND_INTERVAL_SEC", "1.0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))

app = Flask(__name__)
_http = requests.Session()


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


# Настройки 10-минутного (у тебя сейчас тестово 10 секунд)
REPORT_API_URL = os.getenv(
    "REPORT_API_URL",
    "https://shum.i20h.ru/api/v1/measurements/capture/"
)
REPORT_INTERVAL_SEC = int(os.getenv("REPORT_INTERVAL_SEC", "10"))  # для боевого: 600
DEVICE_ID = os.getenv("DEVICE_ID", get_rpi_serial())  # device_serial


# ========= Helpers =========

def db_rows(query: str, args: tuple = ()) -> list[sqlite3.Row]:
    conn = sqlite3.connect("sound_log.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, args)
    rows = cur.fetchall()
    conn.close()
    return rows


def get_last_measurements(limit: int = 20):
    rows = db_rows("SELECT * FROM measurements ORDER BY timestamp DESC LIMIT ?", (limit,))
    if not rows:
        return [], []
    columns = list(rows[0].keys())
    return columns, [tuple(r) for r in rows]


def get_10min_max_level():
    """
    Возвращает (max_leq, ts_at_max) за последние 10 минут/секунд.
    Сейчас окно 10 секунд для теста.
    """
    rows = db_rows(
        """
        SELECT timestamp, leq_1s
        FROM measurements
        WHERE timestamp >= datetime('now', '-10 seconds')
          AND leq_1s IS NOT NULL
        ORDER BY leq_1s DESC
        LIMIT 1
        """
    )
    if not rows:
        return None, None

    row = rows[0]
    return row["leq_1s"], row["timestamp"]


def _to_iso(ts_str: str) -> str:
    """Преобразует 'YYYY-MM-DD HH:MM:SS' -> ISO8601 с Z."""
    try:
        dt = datetime.datetime.fromisoformat(ts_str)
    except ValueError:
        dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    dt_utc = dt.astimezone(datetime.timezone.utc)
    return dt_utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


# ========= 10-минутный отчёт =========

def send_10min_report():
    """
    Формирует и отправляет JSON вида (список!):
    [
      {
        "device_serial": "...",
        "value": <максимальный leq_1s>,
        "event_time": "ISO-время этого максимума"
      }
    ]
    """
    if not REPORT_API_URL:
        print("[REPORT] REPORT_API_URL не задан, отправка отключена")
        return

    max_leq, ts_at_max = get_10min_max_level()
    if max_leq is None or ts_at_max is None:
        print("[REPORT] За окно измерений данных нет, JSON не отправляем")
        return

    payload = [{
        "device_serial": DEVICE_ID,
        "value": float(max_leq),
        "event_time": _to_iso(ts_at_max),
    }]

    try:
        resp = _http.post(
            REPORT_API_URL,
            json=payload,
            timeout=10
        )
        if 200 <= resp.status_code < 300:
            print(f"[REPORT] OK value={max_leq:.2f} dB at {ts_at_max}")
        else:
            print(f"[REPORT] FAIL {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[REPORT] ERROR: {e}")


def report_loop():
    if not REPORT_API_URL:
        print("[REPORT] REPORT_API_URL не задан, репортер не запущен")
        return

    print(f"[REPORT] Старт репортера: интервал {REPORT_INTERVAL_SEC} сек, "
          f"URL={REPORT_API_URL}, device_serial={DEVICE_ID}")
    while True:
        send_10min_report()
        time.sleep(REPORT_INTERVAL_SEC)


def start_reporter():
    t = threading.Thread(target=report_loop, daemon=True)
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

    if not os.path.exists(audio_path):
        print(f"[EVENT] Файл аудио не найден: {audio_path}")
        return

    event_time_iso = _to_iso(event_ts)
    payload = [{
        "device_serial": DEVICE_ID,
        "value": float(value),
        "event_time": event_time_iso,
    }]

    # 1) JSON
    try:
        resp = _http.post(REPORT_API_URL, json=payload, timeout=10)
        if not (200 <= resp.status_code < 300):
            print(f"[EVENT] FAIL JSON {resp.status_code}: {resp.text}")
            return
        try:
            data = resp.json()
        except Exception:
            print(f"[EVENT] Не удалось разобрать JSON ответа: {resp.text}")
            return

        # Ожидаем список объектов, берём первый
        measurement_id = None
        if isinstance(data, list) and data:
            measurement_id = data[0].get("id")
        elif isinstance(data, dict):
            measurement_id = data.get("id") or data.get("measurement")

        if not measurement_id:
            print(f"[EVENT] Не удалось получить id измерения из ответа: {data}")
            return

        measurement_id = int(measurement_id)
        print(f"[EVENT] JSON OK, measurement_id={measurement_id}")
    except Exception as e:
        print(f"[EVENT] ERROR при отправке JSON: {e}")
        return

    # 2) AUDIO
    base = REPORT_API_URL.rstrip("/")
    audio_url = f"{base}/{measurement_id}/audio/"
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            resp2 = _http.post(audio_url, files=files, timeout=30)
        if 200 <= resp2.status_code < 300:
            print(f"[EVENT] AUDIO OK id={measurement_id} file={audio_path}")
        else:
            print(f"[EVENT] AUDIO FAIL {resp2.status_code}: {resp2.text}")
    except Exception as e:
        print(f"[EVENT] ERROR при отправке аудио: {e}")


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
    conn = sqlite3.connect("sound_log.db")
    c = conn.cursor()
    c.execute("SELECT timestamp, spl, leq_1s, lmax FROM measurements ORDER BY timestamp DESC LIMIT 60")
    rows = c.fetchall()
    conn.close()
    data = {
        "timestamps": [r[0] for r in reversed(rows)],
        "spl": [r[1] for r in reversed(rows)],
        "leq": [r[2] for r in reversed(rows)],
        "lmax": [r[3] for r in reversed(rows)],
    }
    return jsonify(data)


@app.route("/api/octave")
def get_latest_octaves():
    conn = sqlite3.connect("sound_log.db")
    c = conn.cursor()
    c.execute("SELECT * FROM measurements ORDER BY timestamp DESC LIMIT 1")
    row = c.fetchone()
    columns = [desc[0] for desc in c.description] if c.description else []
    conn.close()
    if row:
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


if __name__ == "__main__":
    # start_sender()  # не нужен
    start_reporter()
    app.run(host="0.0.0.0", port=5000, debug=True)
