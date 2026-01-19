import os
import json
import time
import threading
from datetime import datetime, timezone
from collections import deque
from queue import Queue, Full, Empty

import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.fft import rfft, rfftfreq

from web_app import app, start_reporter, send_event_with_audio
from state import set_fft
from calibration_utils import load_calibration_curve, apply_frequency_calibration
from calibration import apply_calibration
from weighting import apply_a_weighting
from utils import apply_ema
from octave_analysis import octave_band_levels
from logger import init_csv
from logger_sqlite import init_db, log_to_db
from spl_utils import compute_spl, compute_leq

# ================== CONFIG ==================
DURATION = 1
SAMPLE_RATE = 48000
REFERENCE_PRESSURE = 20e-6

with open("umik_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

sensitivity = float(config.get("sensitivity", 0.0045))
WEIGHTING_MODE = config.get("weighting_mode", "Slow")

# Порог события (превышение) — из конфига/ENV
EVENT_THRESHOLD_DB = float(os.getenv("EVENT_THRESHOLD_DB", config.get("event_threshold_db", 45.0)))
EVENT_PRE_SEC = int(os.getenv("EVENT_PRE_SEC", config.get("event_pre_sec", 15)))
EVENT_POST_SEC = int(os.getenv("EVENT_POST_SEC", config.get("event_post_sec", 15)))
EVENT_END_HOLD_SEC = int(os.getenv("EVENT_END_HOLD_SEC", config.get("event_end_hold_sec", 2)))  # сколько секунд ниже порога считаем концом
EVENT_MIN_SEC = float(os.getenv("EVENT_MIN_SEC", config.get("event_min_sec", 1.0)))

EVENT_OUT_DIR = os.getenv("EVENT_OUT_DIR", config.get("event_out_dir", "public/events"))
os.makedirs(EVENT_OUT_DIR, exist_ok=True)

# Новый endpoint из PDF (noise_raw_data)
NOISE_RAW_API_URL = os.getenv("NOISE_RAW_API_URL", "https://int.kik.mos.ru/noise_raw_data")
NOISE_RAW_ENABLED = os.getenv("NOISE_RAW_ENABLED", "1") == "1"

# Метаданные (из PDF): координаты/УИН/тип сообщения
SERIAL_NUMBER = os.getenv("SERIAL_NUMBER", config.get("serial_number", ""))  # можно пусто, тогда возьмём cpu serial
MESSAGE_TYPE = os.getenv("MESSAGE_TYPE", config.get("message_type", "noise_raw_data"))

LAT = os.getenv("LATITUDE_EQUIP", config.get("latitude_equip", None))
LON = os.getenv("LONGITUDE_EQUIP", config.get("longitude_equip", None))
ALT = os.getenv("ALTITUDE_EQUIP", config.get("altitude_equip", None))

UIN_LIST = os.getenv("UIN_LIST", config.get("uin_list", ""))
UIN = [x.strip() for x in UIN_LIST.split(",") if x.strip()]

# Калибровка UMIK
freqs, gains = load_calibration_curve("7142078_90deg.txt")

# Leq 60 секунд
leq_buffer = deque(maxlen=60)

# ================== HELPERS ==================
def get_rpi_serial() -> str:
    serial = "UNKNOWN"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("Serial"):
                    serial = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    return serial

if not SERIAL_NUMBER:
    SERIAL_NUMBER = get_rpi_serial()

def pick_input_device() -> int:
    """
    Надёжно выбираем устройство ввода.
    1) Если задан UMIK_DEVICE_INDEX — используем его
    2) Иначе пробуем найти по miniDSP/umik
    3) Иначе, если физически один input-девайс (кроме pulse/default) — берём его
    4) Иначе — sd.default.device[0]
    """
    env_idx = os.getenv("UMIK_DEVICE_INDEX")
    if env_idx is not None:
        return int(env_idx)

    devices = sd.query_devices()
    input_idxs = [i for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]

    # 2) поиск по имени
    for i in input_idxs:
        name = (devices[i].get("name") or "").lower()
        if "umik" in name or "minidsp" in name:
            return i

    # 3) если один физический (без pulse/default)
    physical = []
    for i in input_idxs:
        name = (devices[i].get("name") or "").lower()
        if "pulse" in name or name.strip() == "default":
            continue
        physical.append(i)
    if len(physical) == 1:
        return physical[0]

    # 4) default input
    return int(sd.default.device[0])

# ================== NOISE_RAW SENDER ==================
_noise_q: "Queue[dict]" = Queue(maxsize=5)

def _noise_raw_sender_loop():
    import requests
    s = requests.Session()

    while True:
        try:
            payload = _noise_q.get(timeout=1.0)
        except Empty:
            continue

        # отправка
        try:
            r = s.post(
                NOISE_RAW_API_URL,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if not (200 <= r.status_code < 300):
                print(f"[RAW] FAIL {r.status_code}: {r.text}")
        except Exception as e:
            print(f"[RAW] ERROR: {e}")

def start_noise_raw_sender():
    if not NOISE_RAW_ENABLED:
        print("[RAW] disabled")
        return
    if not NOISE_RAW_API_URL:
        print("[RAW] NOISE_RAW_API_URL empty, disabled")
        return
    t = threading.Thread(target=_noise_raw_sender_loop, daemon=True)
    t.start()
    print(f"[RAW] sender started url={NOISE_RAW_API_URL}")

def push_noise_raw(las_value: float):
    """
    Формируем JSON по PDF и кладём в очередь (не блокируем аудио callback).
    """
    now = datetime.now(timezone.utc)
    payload = {
        "serial_number": SERIAL_NUMBER,
        "las": float(las_value),
        "dt": now.isoformat().replace("+00:00", "Z"),
        "time_stamp": int(now.timestamp()),  # unixtime seconds
        "message_type": MESSAGE_TYPE,
    }
    # координаты/высота, если заданы
    if LAT is not None:
        payload["latitude_equip"] = float(LAT)
    if LON is not None:
        payload["longitude_equip"] = float(LON)
    if ALT is not None:
        payload["altitude_equip"] = float(ALT)
    if UIN:
        payload["uin"] = UIN

    try:
        _noise_q.put_nowait(payload)
    except Full:
        # если сеть тормозит — пропускаем, чтобы не убить реальное время
        pass

# ================== EVENT RECORDER ==================
class EventState:
    def __init__(self):
        self.active = False
        self.event_id = 0
        self.start_ts = None
        self.last_above_ts = None
        self.prebuf = deque(maxlen=EVENT_PRE_SEC * SAMPLE_RATE)
        self.blocks = []  # list[np.ndarray] raw mono float32
        self.below_count = 0
        self.filepath = None

event_state = EventState()

def _write_event_wav(path: str, data: np.ndarray):
    sf.write(path, data, SAMPLE_RATE, subtype="PCM_16")

# ================== AUDIO CALLBACK ==================
def audio_callback(indata, frames, time_info, status):
    try:
        if status:
            print(f"[UMIK] Status: {status}")

        mono = indata[:, 0].astype(np.float64)

        # pre-buffer raw for event (float32)
        raw_f32 = mono.astype(np.float32)
        event_state.prebuf.extend(raw_f32)

        # SPL pipeline
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

        leq_buffer.append(weighted)
        all_data = np.concatenate(list(leq_buffer)) if leq_buffer else weighted
        leq_60s = compute_leq(all_data, REFERENCE_PRESSURE)
        lmax = 20 * np.log10(np.max(np.abs(weighted)) / REFERENCE_PRESSURE + 1e-15)

        print(f"SPL: {spl:.1f} dBA | Leq_1s: {leq_1s:.1f} dBA | Leq_60s: {leq_60s:.1f} dBA | Lmax: {lmax:.1f} dBA")

        # 1) запись в БД
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_to_db(timestamp, spl, leq_1s, leq_60s, lmax, bands)

        # 2) новый RAW JSON раз в секунду (мы и так в 1Hz callback)
        if NOISE_RAW_ENABLED:
            push_noise_raw(leq_1s)

        # 3) событие по превышению
        now_t = time.time()
        above = leq_1s >= EVENT_THRESHOLD_DB

        if not event_state.active:
            if above:
                event_state.active = True
                event_state.event_id += 1
                event_state.start_ts = now_t
                event_state.last_above_ts = now_t
                event_state.below_count = 0
                event_state.blocks = []

                ts_name = datetime.now().strftime("%Y%m%dT%H%M%S")
                event_state.filepath = os.path.join(EVENT_OUT_DIR, f"event_{ts_name}.wav")

                print(f"[EVENT] START id={event_state.event_id} file={event_state.filepath} thr={EVENT_THRESHOLD_DB:.1f} dBA")

                # сразу положим prebuf как старт
                pre = np.array(event_state.prebuf, dtype=np.float32)
                if pre.size > 0:
                    event_state.blocks.append(pre)
                event_state.blocks.append(raw_f32.copy())
        else:
            # event active
            event_state.blocks.append(raw_f32.copy())
            if above:
                event_state.last_above_ts = now_t
                event_state.below_count = 0
            else:
                event_state.below_count += 1

            # конец события: N секунд ниже порога
            if event_state.below_count >= EVENT_END_HOLD_SEC:
                duration_sec = now_t - (event_state.start_ts or now_t)
                # добавим post хвост фиксированной длины
                # (post_sec * 1Hz => просто ждём пока callback набежит; проще: набираем EVENT_POST_SEC блоков ниже порога)
                # В этой реализации: раз уже ниже, добираем EVENT_POST_SEC секунд и закрываем.
                # Для простоты: используем below_count как количество секунд ниже порога.
                if event_state.below_count < EVENT_POST_SEC:
                    return

                # минимальная длина события
                if duration_sec < EVENT_MIN_SEC:
                    print(f"[EVENT] DROP too short ({duration_sec:.2f}s)")
                else:
                    wav = np.concatenate(event_state.blocks).astype(np.float32)
                    _write_event_wav(event_state.filepath, wav)
                    print(f"[EVENT] SAVED {event_state.filepath} len={len(wav)/SAMPLE_RATE:.2f}s")

                    # отправка события + аудио (твоя функция в web_app.py)
                    try:
                        send_event_with_audio(
                            event_id=event_state.event_id,
                            wav_path=event_state.filepath,
                            peak_db=float(lmax),
                            leq_db=float(leq_1s),
                            threshold_db=float(EVENT_THRESHOLD_DB),
                            started_at_iso=datetime.fromtimestamp(event_state.start_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                            ended_at_iso=datetime.fromtimestamp(now_t, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                        )
                    except Exception as e:
                        print(f"[EVENT] SEND ERROR: {e}")

                # reset
                event_state.active = False
                event_state.blocks = []
                event_state.start_ts = None
                event_state.last_above_ts = None
                event_state.below_count = 0
                event_state.filepath = None

    except Exception as e:
        print(f"[ERROR] UMIK callback crashed: {e}")

# ================== WEB SERVER ==================
def start_web():
    app.run(host="0.0.0.0", port=5000, debug=False)

# ================== MAIN ==================
if __name__ == "__main__":
    init_csv()
    init_db()

    # репортер (10-минутка)
    start_reporter()

    # RAW sender
    start_noise_raw_sender()

    # веб
    threading.Thread(target=start_web, daemon=True).start()

    dev_index = pick_input_device()
    print(f"[AUDIO] input device index={dev_index} name={sd.query_devices()[dev_index]['name']}")

    with sd.InputStream(
        device=dev_index,
        channels=1,
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLE_RATE * DURATION,
    ):
        print("🎤 Запись с UMIK... (Ctrl+C для выхода)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nЗавершено.")