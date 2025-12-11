import os
import json
import time
import sqlite3
import threading
from collections import deque
from pathlib import Path
from datetime import datetime, time as dtime
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.fft import rfft, rfftfreq
import requests

from web_app import app
from state import set_fft
from calibration_utils import load_calibration_curve, apply_frequency_calibration
from calibration import apply_calibration
from weighting import apply_a_weighting
from utils import apply_ema
from octave_analysis import octave_band_levels
from logger import init_csv, log_to_csv
from logger_sqlite import init_db, log_to_db, insert_event_start, update_event_end
from spl_utils import compute_spl, compute_leq


DURATION = 1
SAMPLE_RATE = 48000
REFERENCE_PRESSURE = 20e-6

with open('umik_config.json') as f:
    config = json.load(f)

sensitivity = float(config.get('sensitivity', 0.0045))
WEIGHTING_MODE = config.get('weighting_mode', 'Slow')

ANALOG_DEVICE_SUBSTR = os.getenv('ANALOG_DEVICE_SUBSTR', config.get('analog_device_substr', '')) 
ANALOG_SR        = int(os.getenv('ANALOG_SR',        config.get('analog_sr', SAMPLE_RATE)))
ANALOG_CH        = int(os.getenv('ANALOG_CHANNELS',  config.get('analog_channels', 1)))
ANALOG_BLOCK     = int(os.getenv('ANALOG_BLOCK',     config.get('analog_block', 2048)))
SEGMENT_SEC      = int(os.getenv('ANALOG_SEGMENT_SEC', config.get('analog_segment_sec', 300)))  
OUT_PATH         = Path(os.getenv('ANALOG_OUT_PATH', config.get('analog_out_path', 'public/analog.wav')))

freqs, gains = load_calibration_curve("7142078_90deg.txt")

leq_buffer = deque(maxlen=60)

DAY_THRESHOLD = 55.0    
NIGHT_THRESHOLD = 45.0  

PRE_EVENT_SEC = 15           
POST_EVENT_SEC = 15          
AUDIO_EVENTS_DIR = Path("public/events")
AUDIO_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

pre_event_buffer = deque(maxlen=PRE_EVENT_SEC) 
event_recording = False
event_post_left = 0          
event_writer = None          
event_max_leq = 0.0
current_event_id = None
current_event_threshold = None
current_event_audio_path = None  

LAST_ID_FILE = "last_capture_id.txt"
SERVER_BASE = "https://shum.i20h.ru/api/v1"
session = requests.Session()



def get_current_threshold(now=None) -> float:
    if now is None:
        now = datetime.now()
    t = now.time()
    if dtime(7, 0) <= t < dtime(23, 0):
        return DAY_THRESHOLD
    else:
        return NIGHT_THRESHOLD


def get_last_capture_id() -> int | None:

    try:
        with open(LAST_ID_FILE, "r") as f:
            s = f.read().strip()
            if not s:
                return None
            return int(s)
    except Exception:
        return None


def send_audio_for_last_capture(audio_path: str):

    measurement_id = get_last_capture_id()
    if measurement_id is None:
        print("[EVENT->API] Нет сохранённого id 10-минутного измерения, аудио отправить нельзя")
        return

    audio_url = f"{SERVER_BASE}/measurements/capture/{measurement_id}/audio/"
    print(f"[EVENT->API] measurement id={measurement_id}, отправляем аудио {audio_path}")

    try:
        with open(audio_path, "rb") as f:
            files = {
                "audio": (os.path.basename(audio_path), f, "audio/wav"),
            }
            resp = session.post(audio_url, files=files, timeout=60)

        if 200 <= resp.status_code < 300:
            print(f"[EVENT->API] OK audio uploaded for measurement {measurement_id}")
        else:
            print(f"[EVENT->API] FAIL audio upload: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[EVENT->API] ERROR audio upload: {e}")


def start_noise_event(now: datetime, threshold: float):
    global event_recording, event_writer, event_post_left, event_max_leq
    global current_event_id, current_event_threshold, current_event_audio_path

    event_recording = True
    event_post_left = POST_EVENT_SEC
    event_max_leq = 0.0
    current_event_threshold = threshold

    ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
    filename = f"event_{now.strftime('%Y%m%dT%H%M%S')}.wav"
    filepath = AUDIO_EVENTS_DIR / filename
    current_event_audio_path = str(filepath)

    event_writer = sf.SoundFile(
        str(filepath),
        mode="w",
        samplerate=SAMPLE_RATE,
        channels=1,
        subtype="PCM_16"
    )

    for chunk in pre_event_buffer:
        event_writer.write(chunk)

    current_event_id = insert_event_start(ts_str, threshold, str(filepath))
    print(f"[EVENT] START id={current_event_id} file={filepath} thr={threshold:.1f} dBA")


def stop_noise_event(now: datetime):
    global event_recording, event_writer, current_event_id, event_max_leq
    global current_event_threshold, current_event_audio_path

    if not event_recording:
        return

    event_recording = False
    ts_str = now.strftime("%Y-%m-%d %H:%M:%S")

    if event_writer is not None:
        event_writer.close()
        event_writer = None

    if current_event_id is not None:
        update_event_end(current_event_id, ts_str, event_max_leq)
        print(f"[EVENT] STOP id={current_event_id} max_leq={event_max_leq:.1f} thr={current_event_threshold:.1f} at {ts_str}")

        if current_event_audio_path is not None:
            send_audio_for_last_capture(current_event_audio_path)

        current_event_id = None
        current_event_threshold = None
        current_event_audio_path = None


def init_weighted_table():
    with sqlite3.connect("sound_log.db") as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS weighted_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                weight_type TEXT,
                spl REAL,
                leq REAL,
                lmax REAL
            )
        """)
        conn.commit()


def find_input_device(substr: str) -> int | None:
    if not substr:
        return None
    s = substr.lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0 and s in dev['name'].lower():
            return i
    return None


def get_umick_index():
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            print(f"[UMIK] Берём устройство #{i}: {dev['name']}")
            return i
    raise RuntimeError("UMIK-1 не найден")



def audio_callback(indata, frames, time_info, status):
    global event_recording, event_post_left, event_max_leq, event_writer

    try:
        if status:
            print(f"[UMIK] Status: {status}")

        mono = indata[:, 0].astype(np.float64)

        pre_event_buffer.append(mono.copy())

        now = datetime.now()
        threshold = get_current_threshold(now)

        pressure_signal = apply_calibration(mono, sensitivity)
        pressure_signal = apply_frequency_calibration(pressure_signal, SAMPLE_RATE, freqs, gains)

        fft_result = np.abs(rfft(pressure_signal))
        fft_freqs = rfftfreq(len(pressure_signal), d=1 / SAMPLE_RATE)
        mask = fft_freqs <= 20000
        set_fft({"freqs": fft_freqs[mask].tolist(), "values": fft_result[mask].tolist()})

        weighted = apply_a_weighting(pressure_signal, SAMPLE_RATE)

        if WEIGHTING_MODE == "Fast":
            weighted = apply_ema(weighted, alpha=0.125)
        elif WEIGHTING_MODE == "Slow":
            weighted = apply_ema(weighted, alpha=0.03125)

        spl = compute_spl(weighted, REFERENCE_PRESSURE)
        leq_1s = compute_leq(weighted, REFERENCE_PRESSURE)

        bands = octave_band_levels(weighted, SAMPLE_RATE)
        print(f"Octaves dBA: {bands}")

        leq_buffer.append(weighted)
        all_data = np.concatenate(list(leq_buffer)) if leq_buffer else weighted
        leq_60s = compute_leq(all_data, REFERENCE_PRESSURE)
        lmax = 20 * np.log10(np.max(np.abs(weighted)) / REFERENCE_PRESSURE + 1e-15)

        print(f"SPL: {spl:.1f} dBA | Leq_1s: {leq_1s:.1f} dBA | Leq_60s: {leq_60s:.1f} dBA | Lmax: {lmax:.1f} dBA")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_to_db(timestamp, spl, leq_1s, leq_60s, lmax, bands)

        is_exceed = (leq_1s is not None) and (leq_1s > threshold)

        if event_recording and event_writer is not None:
            event_writer.write(mono)
            if leq_1s is not None and leq_1s > event_max_leq:
                event_max_leq = leq_1s

        if event_recording:
            if is_exceed:
                event_post_left = POST_EVENT_SEC
            else:
                event_post_left -= 1
                if event_post_left <= 0:
                    stop_noise_event(now)
        else:
            if is_exceed:
                start_noise_event(now, threshold)
                if event_writer is not None:
                    event_writer.write(mono)
                event_max_leq = leq_1s if leq_1s is not None else 0.0

    except Exception as e:
        print(f"[ERROR] UMIK callback crashed: {e}")



def start_analog_recorder():
    try:
        idx = find_input_device(ANALOG_DEVICE_SUBSTR)
        if idx is None:
            print("[ANALOG] Устройство не найдено. Запись аналога отключена.")
            return

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        q: Queue[np.ndarray] = Queue(maxsize=50)

        def analog_cb(indata, frames, time_info, status):
            if status:
                print(f"[ANALOG] Status: {status}")
            q.put(indata.copy())

        print(f"[ANALOG] Используется устройство: {sd.query_devices()[idx]['name']} (idx={idx})")
        with sd.InputStream(device=idx, channels=ANALOG_CH, samplerate=ANALOG_SR,
                            blocksize=ANALOG_BLOCK, callback=analog_cb, dtype='float32'):
            print("🎙️ Аналоговый микрофон: запись по 5 минут с перезаписью файла.")
            while True:
                with sf.SoundFile(str(OUT_PATH), mode='w', samplerate=ANALOG_SR,
                                  channels=ANALOG_CH, subtype='PCM_16') as wav:
                    t_end = time.time() + SEGMENT_SEC
                    while time.time() < t_end:
                        try:
                            block = q.get(timeout=0.5)
                            wav.write(block)
                        except Empty:
                            pass
                print(f"[ANALOG] Сегмент {SEGMENT_SEC}s записан → {OUT_PATH.name} (перезапись началась заново)")

    except Exception as e:
        print(f"[ERROR] Аналоговый рекордер не запустился: {e}")



def start_web():
    if not os.path.exists("templates/table.html"):
        print("[WARNING] Шаблон templates/table.html не найден!")
    app.run(host="0.0.0.0", port=5000, debug=False)



if __name__ == "__main__":
    init_csv()
    init_db()
    init_weighted_table()

    threading.Thread(target=start_web, daemon=True).start()

    if ANALOG_DEVICE_SUBSTR:
        threading.Thread(target=start_analog_recorder, daemon=True).start()
    else:
        print("[ANALOG] analog_device_substr пуст — аналоговая запись отключена.")

    with sd.InputStream(
        device=get_umick_index(),
        channels=1,
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLE_RATE * DURATION
    ):
        print("🎤 Запись с UMIK-1... (нажмите Ctrl+C для выхода)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nЗавершено.")
