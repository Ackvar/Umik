# web_app.py
from __future__ import annotations
from flask import Flask, render_template, jsonify, request
import sqlite3
import os
import time
import json
import threading
import requests
from state import get_fft

# ========= Настройки внешней отправки (старый батчевый экспорт, если нужен) =========
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL")         # напр. http://185.231.153.2:8080/ingest
EXTERNAL_API_TOKEN = os.getenv("EXTERNAL_API_TOKEN")     # если нужен
SEND_INTERVAL_SEC = float(os.getenv("SEND_INTERVAL_SEC", "1.0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))

app = Flask(__name__)
_http = requests.Session()

# Флаг: когда удачно ушёл 10-минутный отчёт, нужно обрезать текущий аудио-ивент
force_cut_event = False


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


# ========= Настройки 10-минутного отчёта =========
REPORT_API_URL = os.getenv(
    "REPORT_API_URL",
    "https://shum.i20h.ru/api/v1/measurements/capture/"
)
# Для теста 10 секунд, потом поставишь 600
REPORT_INTERVAL_SEC = int(os.getenv("REPORT_INTERVAL_SEC", "10"))
DEVICE_ID = os.getenv("DEVICE_ID", get_rpi_serial())


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
    Возвращает (max_leq, ts_at_max) за последние 10 минут.
    max_leq: максимальный LAeq (leq_1s) в дБ(A)
    ts_at_max: timestamp, когда он был измерен.
    """
    rows = db_rows(
        """
        SELECT timestamp, leq_1s
        FROM measurements
        WHERE timestamp >= datetime('now', '-600 seconds')
          AND leq_1s IS NOT NULL
        ORDER BY leq_1s DESC
        LIMIT 1
        """
    )
    if not rows:
        return None, None

    row = rows[0]
    return row["leq_1s"], row["timestamp"]


def send_10min_report():
    """
    Отправляет JSON-список:
    [
      {
        "device_serial": "...",
        "value": <максимальный Leq_1s за 10 минут>,
        "event_time": "YYYY-MM-DD HH:MM:SS"
      }
    ]
    При успешной отправке поднимает флаг force_cut_event,
    чтобы main.py обрезал текущий аудио-ивент.
    """
    if not REPORT_API_URL:
        print("[REPORT] REPORT_API_URL не задан, отправка отключена")
        return

    max_leq, ts_at_max = get_10min_max_level()
    if max_leq is None or ts_at_max is None:
        print("[REPORT] За последние 10 минут измерений нет, JSON не отправляем")
        return

    payload = [{
        "device_serial": DEVICE_ID,
        "value": float(max_leq),
        "event_time": ts_at_max
    }]

    global force_cut_event

    try:
        resp = _http.post(
            REPORT_API_URL,
            json=payload,   # отправляем список объектов
            timeout=10
        )
        if 200 <= resp.status_code < 300:
            print(f"[REPORT] OK value={max_leq:.2f} dB at {ts_at_max}")
            # ставим флаг: пора обрезать текущий ивент (если он есть)
            force_cut_event = True
        else:
            print(f"[REPORT] FAIL {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[REPORT] ERROR: {e}")


def report_loop():
    if not REPORT_API_URL:
        print("[REPORT] REPORT_API_URL не задан, репортер не запущен")
        return

    print(f"[REPORT] Старт репортера: интервал {REPORT_INTERVAL_SEC} сек, URL={REPORT_API_URL}, device_serial={DEVICE_ID}")
    while True:
        send_10min_report()
        time.sleep(REPORT_INTERVAL_SEC)


def start_reporter():
    t = threading.Thread(target=report_loop, daemon=True)
    t.start()

# ========= Pages =========

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

# ========= APIs =========

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
            "31.5 Hz","63.0 Hz","125.0 Hz","250.0 Hz",
            "500.0 Hz","1000.0 Hz","2000.0 Hz","4000.0 Hz","8000.0 Hz",
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
        "63":   row.get(keys[-8], None),
        "125":  row.get(keys[-7], None),
        "250":  row.get(keys[-6], None),
        "500":  row.get(keys[-5], None),
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


# ========= Старый батчевый экспорт (если понадобится) =========

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
                    resp = _http.post(EXTERNAL_API_URL, data=json.dumps(payload), headers=headers, timeout=10)
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
