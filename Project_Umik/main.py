import os
import glob
import json
import time
import threading
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from collections import deque
from queue import Queue, Full, Empty

import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.fft import rfft, rfftfreq

from web_app import app, start_reporter
from state import set_fft
from calibration_utils import load_calibration_curve, apply_frequency_calibration
from calibration import apply_calibration
from weighting import apply_a_weighting
from utils import apply_ema
from octave_analysis import octave_band_levels
from logger import init_csv
from logger_sqlite import init_db, log_to_db
from spl_utils import compute_spl, compute_leq
from time_utils import APP_TIMEZONE, app_now, format_db_timestamp, parse_db_timestamp

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
ANALOG_OUT_PATH = os.getenv("ANALOG_OUT_PATH", config.get("analog_out_path", ""))
AUDIO_CLEANUP_INTERVAL_SEC = int(os.getenv("AUDIO_CLEANUP_INTERVAL_SEC", "300"))
AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".flac", ".m4a", ".aac"}
EVENT_AUDIO_FORMAT = os.getenv("EVENT_AUDIO_FORMAT", "aac").strip().lower()
EVENT_AUDIO_BITRATE = os.getenv("EVENT_AUDIO_BITRATE", "96k").strip() or "96k"
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"

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


def _resolve_ffmpeg_bin() -> str | None:
    if os.path.sep in FFMPEG_BIN or (os.path.altsep and os.path.altsep in FFMPEG_BIN):
        return FFMPEG_BIN if os.path.exists(FFMPEG_BIN) else None

    found = shutil.which(FFMPEG_BIN)
    if found:
        return found

    localappdata = os.getenv("LOCALAPPDATA", "")
    userprofile = os.getenv("USERPROFILE", "")
    candidate_patterns = [
        os.path.join(localappdata, "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
        os.path.join(localappdata, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg_*", "**", "bin", "ffmpeg.exe"),
        os.path.join(userprofile, "scoop", "apps", "ffmpeg", "current", "bin", "ffmpeg.exe"),
    ]
    for pattern in candidate_patterns:
        for candidate in glob.glob(pattern, recursive=True):
            if os.path.isfile(candidate):
                return candidate
    return None


def _event_audio_extension() -> str:
    return ".m4a" if EVENT_AUDIO_FORMAT == "aac" else ".wav"


def _encode_aac_file(src_wav_path: str, out_path: str):
    ffmpeg_bin = _resolve_ffmpeg_bin()
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found; install ffmpeg to save event audio in AAC")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src_wav_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "aac",
        "-b:a",
        EVENT_AUDIO_BITRATE,
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"ffmpeg AAC encode failed: {message}")


def _write_event_audio(path: str, data: np.ndarray):
    if EVENT_AUDIO_FORMAT != "aac":
        sf.write(path, data, SAMPLE_RATE, subtype="PCM_16")
        return

    temp_wav_path = f"{os.path.splitext(path)[0]}.tmp.wav"
    try:
        sf.write(temp_wav_path, data, SAMPLE_RATE, subtype="PCM_16")
        _encode_aac_file(temp_wav_path, path)
    finally:
        if os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass


def log_audio_encoder_status():
    if EVENT_AUDIO_FORMAT == "aac":
        ffmpeg_bin = _resolve_ffmpeg_bin()
        if ffmpeg_bin:
            print(f"[AUDIO] event format=AAC (.m4a), bitrate={EVENT_AUDIO_BITRATE}, ffmpeg={ffmpeg_bin}")
        else:
            print("[AUDIO] event format=AAC (.m4a), but ffmpeg was not found")
    else:
        print("[AUDIO] event format=WAV")


def _remove_audio_file(path: str) -> bool:
    try:
        os.remove(path)
        print(f"[AUDIO_CLEANUP] deleted {path}")
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[AUDIO_CLEANUP] failed to delete {path}: {e}")
        return False


def cleanup_old_audio_files() -> int:
    today = app_now().date()
    deleted = 0

    for entry in os.scandir(EVENT_OUT_DIR):
        if not entry.is_file():
            continue
        _, ext = os.path.splitext(entry.name)
        if ext.lower() not in AUDIO_EXTENSIONS:
            continue

        file_day = datetime.fromtimestamp(entry.stat().st_mtime, tz=APP_TIMEZONE).date()
        if file_day < today and _remove_audio_file(entry.path):
            deleted += 1

    if ANALOG_OUT_PATH and os.path.isfile(ANALOG_OUT_PATH):
        _, ext = os.path.splitext(ANALOG_OUT_PATH)
        if ext.lower() in AUDIO_EXTENSIONS:
            file_day = datetime.fromtimestamp(os.path.getmtime(ANALOG_OUT_PATH), tz=APP_TIMEZONE).date()
            if file_day < today and _remove_audio_file(ANALOG_OUT_PATH):
                deleted += 1

    if deleted:
        print(f"[AUDIO_CLEANUP] deleted {deleted} old audio file(s)")

    return deleted


def audio_cleanup_loop():
    print(f"[AUDIO_CLEANUP] started, interval={AUDIO_CLEANUP_INTERVAL_SEC}s")
    while True:
        cleanup_old_audio_files()
        time.sleep(AUDIO_CLEANUP_INTERVAL_SEC)


def start_audio_cleanup():
    t = threading.Thread(target=audio_cleanup_loop, daemon=True)
    t.start()

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
        self.lock = threading.RLock()
        self.active = False
        self.event_id = 0
        self.segment_id = 0
        self.start_ts = None
        self.last_above_ts = None
        self.segment_start_dt = None
        self.prebuf = deque(maxlen=max(0, EVENT_PRE_SEC))
        self.blocks = []  # list[(start_dt, end_dt, raw mono float32)]
        self.below_count = 0
        self.filepath = None
        self.max_leq = None
        self.max_ts = None

event_state = EventState()
completed_event_segments = deque(maxlen=200)
event_save_lock = threading.RLock()


def _make_event_audio_path(event_id: int, segment_id: int) -> str:
    ts_name = app_now().strftime("%Y%m%dT%H%M%S")
    return os.path.join(
        EVENT_OUT_DIR,
        f"event_{ts_name}_{event_id:04d}_{segment_id:02d}{_event_audio_extension()}",
    )


def _open_event_segment_locked(segment_start_dt: datetime, event_start_ts: float | None = None):
    event_state.segment_id += 1
    event_state.segment_start_dt = segment_start_dt
    event_state.start_ts = event_start_ts if event_start_ts is not None else segment_start_dt.timestamp()
    event_state.blocks = []
    event_state.filepath = _make_event_audio_path(event_state.event_id, event_state.segment_id)
    event_state.max_leq = None
    event_state.max_ts = None


def _reset_event_locked():
    event_state.active = False
    event_state.blocks = []
    event_state.start_ts = None
    event_state.last_above_ts = None
    event_state.segment_start_dt = None
    event_state.below_count = 0
    event_state.filepath = None
    event_state.max_leq = None
    event_state.max_ts = None


def _update_segment_peak_locked(leq_1s: float, timestamp: str, above: bool):
    if not above:
        return
    if event_state.max_leq is None or leq_1s > event_state.max_leq:
        event_state.max_leq = float(leq_1s)
        event_state.max_ts = timestamp


def _slice_audio_block(block, window_start: datetime | None = None, window_end: datetime | None = None):
    block_start, block_end, data = block[:3]
    clip_start = block_start if window_start is None or block_start >= window_start else window_start
    clip_end = block_end if window_end is None or block_end <= window_end else window_end
    if clip_end <= clip_start:
        return None

    block_seconds = (block_end - block_start).total_seconds()
    if block_seconds <= 0:
        return None

    start_index = int(round((clip_start - block_start).total_seconds() * SAMPLE_RATE))
    end_index = int(round((clip_end - block_start).total_seconds() * SAMPLE_RATE))
    start_index = max(0, min(len(data), start_index))
    end_index = max(0, min(len(data), end_index))
    if end_index <= start_index:
        return None

    return (clip_start, clip_end, data[start_index:end_index].copy(), *block[3:])


def _segment_peak_from_blocks(blocks, window_start: datetime | None = None, window_end: datetime | None = None):
    max_leq = None
    max_ts = None

    for block in blocks:
        if len(block) < 6:
            continue
        _, _, _, leq_1s, timestamp, above = block
        if not above:
            continue

        measurement_dt = parse_db_timestamp(timestamp)
        if window_start is not None and measurement_dt < window_start:
            continue
        if window_end is not None and measurement_dt >= window_end:
            continue

        if max_leq is None or leq_1s > max_leq:
            max_leq = float(leq_1s)
            max_ts = timestamp

    return max_leq, max_ts


def _collect_segment_audio(blocks, window_start: datetime | None = None, window_end: datetime | None = None):
    clipped = []
    chunks = []
    for block in blocks:
        sliced = _slice_audio_block(block, window_start, window_end)
        if not sliced:
            continue
        clipped.append(sliced)
        chunks.append(sliced[2])

    if not chunks:
        return None

    return {
        "audio": np.concatenate(chunks).astype(np.float32),
        "start_dt": clipped[0][0],
        "end_dt": clipped[-1][1],
    }


def _snapshot_event_segment_locked(window_start: datetime | None = None, window_end: datetime | None = None):
    collected = _collect_segment_audio(event_state.blocks, window_start, window_end)
    if not collected:
        return None

    peak_window_start = window_start
    if event_state.start_ts is not None:
        event_start_dt = datetime.fromtimestamp(event_state.start_ts, tz=APP_TIMEZONE)
        if peak_window_start is None or peak_window_start < event_start_dt:
            peak_window_start = event_start_dt
    max_leq, max_ts = _segment_peak_from_blocks(event_state.blocks, peak_window_start, window_end)

    return {
        **collected,
        "audio_path": event_state.filepath or _make_event_audio_path(event_state.event_id, event_state.segment_id),
        "max_leq": max_leq,
        "max_ts": max_ts,
        "has_exceedance": max_leq is not None,
    }


def _prune_completed_segments_locked(cutoff_dt: datetime):
    while completed_event_segments and completed_event_segments[0]["end_dt"] < cutoff_dt:
        completed_event_segments.popleft()


def _save_completed_event_segment(segment: dict | None, reason: str):
    if not segment:
        return None

    audio = segment.pop("audio", None)
    if audio is None or len(audio) == 0:
        return None

    with event_save_lock:
        audio_path = segment["audio_path"]
        try:
            _write_event_audio(audio_path, audio)
        except Exception as e:
            print(f"[EVENT] SAVE ERROR ({reason}): {e}")
            return None

        duration_sec = len(audio) / SAMPLE_RATE
        with event_state.lock:
            completed_event_segments.append(segment)
            _prune_completed_segments_locked(segment["end_dt"] - timedelta(hours=1))

        peak = f"{segment['max_leq']:.2f}" if segment.get("max_leq") is not None else "n/a"
        print(f"[EVENT] SAVED {audio_path} reason={reason} len={duration_sec:.2f}s peak={peak}")
        return audio_path


def close_report_window_audio(window_start: datetime, window_end: datetime):
    segment = None
    with event_state.lock:
        if not event_state.active or not event_state.blocks:
            return

        segment = _snapshot_event_segment_locked(window_start, window_end)
        remaining_blocks = []
        for block in event_state.blocks:
            sliced = _slice_audio_block(block, window_start=window_end)
            if sliced:
                remaining_blocks.append(sliced)

        event_state.segment_id += 1
        event_state.segment_start_dt = window_end
        event_state.start_ts = window_end.timestamp()
        event_state.blocks = remaining_blocks
        event_state.filepath = _make_event_audio_path(event_state.event_id, event_state.segment_id)
        event_state.max_leq = None
        event_state.max_ts = None

        print(
            f"[EVENT] SPLIT window={format_db_timestamp(window_start)}.."
            f"{format_db_timestamp(window_end)} next_file={event_state.filepath}"
        )

    _save_completed_event_segment(segment, "WINDOW")


def find_report_audio(selected: dict):
    if not selected.get("exceeded"):
        return None

    event_dt = parse_db_timestamp(selected["timestamp"])
    with event_save_lock:
        with event_state.lock:
            window_start = selected.get("window_start") or event_dt
            _prune_completed_segments_locked(window_start - timedelta(hours=1))

            candidates = [
                segment for segment in completed_event_segments
                if segment.get("has_exceedance")
                and segment["start_dt"] <= event_dt <= segment["end_dt"]
                and os.path.exists(segment["audio_path"])
            ]

            if not candidates:
                print(f"[REPORT] AUDIO not found for exceeded measurement at {selected['timestamp']}")
                return None

            best = max(candidates, key=lambda segment: segment.get("max_leq") or -float("inf"))
            return best["audio_path"]

# ================== AUDIO CALLBACK ==================
def audio_callback(indata, frames, time_info, status):
    try:
        if status:
            print(f"[UMIK] Status: {status}")

        mono = indata[:, 0].astype(np.float64)

        raw_f32 = mono.astype(np.float32)
        block_end_dt = app_now()
        block_start_dt = block_end_dt - timedelta(seconds=frames / SAMPLE_RATE)
        audio_samples = raw_f32.copy()

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
        timestamp = format_db_timestamp(block_end_dt)
        log_to_db(timestamp, spl, leq_1s, leq_60s, lmax, bands)

        # 2) новый RAW JSON раз в секунду (мы и так в 1Hz callback)
        if NOISE_RAW_ENABLED:
            push_noise_raw(leq_1s)

        # 3) событие по превышению
        now_t = block_end_dt.timestamp()
        above = leq_1s >= EVENT_THRESHOLD_DB
        audio_block = (block_start_dt, block_end_dt, audio_samples, float(leq_1s), timestamp, above)
        segment_to_save = None

        with event_state.lock:
            if not event_state.active:
                if above:
                    event_state.active = True
                    event_state.event_id += 1
                    event_state.segment_id = 0
                    event_state.last_above_ts = now_t
                    event_state.below_count = 0

                    segment_start_dt = event_state.prebuf[0][0] if event_state.prebuf else block_start_dt
                    _open_event_segment_locked(segment_start_dt, event_start_ts=now_t)
                    event_state.blocks = list(event_state.prebuf) + [audio_block]
                    _update_segment_peak_locked(leq_1s, timestamp, above)

                    print(
                        f"[EVENT] START id={event_state.event_id} "
                        f"file={event_state.filepath} thr={EVENT_THRESHOLD_DB:.1f} dBA"
                    )
            else:
                event_state.blocks.append(audio_block)
                if above:
                    event_state.last_above_ts = now_t
                    event_state.below_count = 0
                else:
                    event_state.below_count += 1

                _update_segment_peak_locked(leq_1s, timestamp, above)

                post_blocks_needed = max(EVENT_END_HOLD_SEC, EVENT_POST_SEC)
                if event_state.below_count >= post_blocks_needed:
                    duration_sec = now_t - (event_state.start_ts or now_t)
                    if duration_sec < EVENT_MIN_SEC:
                        print(f"[EVENT] DROP too short ({duration_sec:.2f}s)")
                    else:
                        segment_to_save = _snapshot_event_segment_locked()

                    _reset_event_locked()

            event_state.prebuf.append(audio_block)

        _save_completed_event_segment(segment_to_save, "END")

    except Exception as e:
        print(f"[ERROR] UMIK callback crashed: {e}")

# ================== WEB SERVER ==================
def start_web():
    app.run(host="0.0.0.0", port=5000, debug=False)

# ================== MAIN ==================
if __name__ == "__main__":
    init_csv()
    init_db()

    # репортер capture-отчёта
    start_reporter(
        on_window_close=close_report_window_audio,
        audio_lookup=find_report_audio,
    )

    # статус кодировщика event-аудио
    log_audio_encoder_status()

    # ежедневная очистка старых аудиофайлов
    start_audio_cleanup()

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
