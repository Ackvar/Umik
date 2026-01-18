# main.py
import os
import json
import time
import threading
from collections import deque
from pathlib import Path
from queue import Queue

import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.fft import rfft, rfftfreq
from datetime import datetime, time as dtime

from web_app import app, start_reporter, send_event_with_audio
from state import set_fft
from calibration_utils import load_calibration_curve, apply_frequency_calibration
from calibration import apply_calibration
from weighting import apply_a_weighting
from utils import apply_ema
from octave_analysis import octave_band_levels
from logger import init_csv
from logger_sqlite import init_db, log_to_db, init_db, insert_event_start, update_event_end
from spl_utils import compute_spl, compute_leq

# === Конфигурация UMIK ===
DURATION = 1
SAMPLE_RATE = 48000
REFERENCE_PRESSURE = 20e-6

with open('umik_config.json') as f:
    config = json.load(f)

sensitivity = float(config.get('sensitivity', 0.0045))
WEIGHTING_MODE = config.get('weighting_mode', 'Slow')

# === Калибровка для UMIK ===
freqs, gains = load_calibration_curve("7142078_90deg.txt")

# === Буфер Leq для UMIK (60 последних секунд) ===
leq_buffer = deque(maxlen=60)

# === Пороговые значения ===
DAY_THRESHOLD = 55.0    # 07:00–23:00
NIGHT_THRESHOLD = 45.0  # 23:00–07:00

# === Аудио события ===
PRE_EVENT_SEC = 15        # секунд ДО превышения
POST_EVENT_SEC = 15       # секунд ПОСЛЕ окончания
AUDIO_EVENTS_DIR = Path("public/events")
AUDIO_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

pre_event_buffer = deque(maxlen=PRE_EVENT_SEC)

event_recording = False
event_post_left = 0
event_writer: sf.SoundFile | None = None
event_max_leq = 0.0
event_max_ts: str | None = None
current_event_id: int | None = None
current_event_threshold: float | None = None
current_event_path: str | None = None

# Очередь на отправку события (чтобы не блокировать callback)
upload_queue: Queue[tuple[float, str, str]] = Queue()


def get_current_threshold(now: datetime | None = None) -> float:
    if now is None:
        now = datetime.now()
    t = now.time()
    if dtime(7, 0) <= t < dtime(23, 0):
        return DAY_THRESHOLD
    return NIGHT_THRESHOLD


def get_umick_index() -> int:
    for i, dev in enumerate(sd.query_devices()):
        name = (dev.get("name") or "").lower()
        if dev.get("max_input_channels", 0) > 0:
            if name.startswith("pulse") or name.startswith("default"):
                continue
            return i
    raise RuntimeError("Не найдено ни одного физического устройства ввода (микрофона)")


def start_noise_event(now: datetime, threshold: float):
    """Старт события: создаём WAV, пишем префикс, запись в БД."""
    global event_recording, event_post_left, event_writer
    global event_max_leq, event_max_ts, current_event_id, current_event_threshold, current_event_path

    event_recording = True
    event_post_left = POST_EVENT_SEC
    event_max_leq = 0.0
    event_max_ts = None
    current_event_threshold = threshold

    ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
    filename = f"event_{now.strftime('%Y%m%dT%H%M%S')}.wav"
    filepath = AUDIO_EVENTS_DIR / filename
    current_event_path = str(filepath)

    event_writer = sf.SoundFile(
        str(filepath),
        mode="w",
        samplerate=SAMPLE_RATE,
        channels=1,
        subtype="PCM_16",
    )

    # записываем прелоад (15 секунд до события)
    for chunk in pre_event_buffer:
        event_writer.write(chunk)

    current_event_id = insert_event_start(ts_str, threshold, str(filepath))
    print(f"[EVENT] START id={current_event_id} file={filepath} thr={threshold:.1f} dBA")


def stop_noise_event(now: datetime):
    """Завершение события: закрываем файл, обновляем БД и ставим отправку в очередь."""
    global event_recording, event_writer, current_event_id
    global event_max_leq, event_max_ts, current_event_threshold, current_event_path

    if not event_recording:
        return

    event_recording = False
    ts_str = now.strftime("%Y-%m-%d %H:%M:%S")

    if event_writer is not None:
        event_writer.close()
        event_writer = None

    if current_event_id is not None:
        update_event_end(current_event_id, ts_str, event_max_leq)
        print(f"[EVENT] STOP id={current_event_id} max_leq={event_max_leq:.1f} "
              f"thr={current_event_threshold:.1f} at {ts_str}")

    # ставим отправку в очередь (если есть аудио и хоть какой-то уровень)
    if current_event_path and event_max_leq > 0.0:
        upload_queue.put((event_max_leq, event_max_ts or ts_str, current_event_path))

    current_event_id = None
    current_event_threshold = None
    current_event_path = None
    event_max_leq = 0.0
    event_max_ts = None


def uploader_worker():
    """Фоновая отправка JSON+audio для завершённых событий."""
    while True:
        value, ts_str, audio_path = upload_queue.get()
        try:
            print(f"[EVENT] UPLOAD queued value={value:.1f} ts={ts_str} file={audio_path}")
            send_event_with_audio(value, ts_str, audio_path)
        except Exception as e:
            print(f"[EVENT] UPLOAD ERROR: {e}")


def audio_callback(indata, frames, time_info, status):
    global event_recording, event_post_left, event_writer
    global event_max_leq, event_max_ts

    try:
        if status:
            print(f"[UMIK] Status: {status}")

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        threshold = get_current_threshold(now)

        mono = indata[:, 0].astype(np.float64)

        # буфер для 15с до события
        pre_event_buffer.append(mono.copy())

        pressure_signal = apply_calibration(mono, sensitivity)
        pressure_signal = apply_frequency_calibration(pressure_signal, SAMPLE_RATE, freqs, gains)

        # FFT → веб
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

        print(f"SPL: {spl:.1f} dBA | Leq_1s: {leq_1s:.1f} dBA | "
              f"Leq_60s: {leq_60s:.1f} dBA | Lmax: {lmax:.1f} dBA")

        # логируем в БД
        log_to_db(timestamp, spl, leq_1s, leq_60s, lmax, bands)

        # превышение порога?
        is_exceed = leq_1s is not None and leq_1s > threshold

        # если сейчас пишем событие — дозаписываем блок и обновляем максимум
        if event_recording and event_writer is not None:
            event_writer.write(mono)
            if leq_1s is not None and leq_1s > event_max_leq:
                event_max_leq = leq_1s
                event_max_ts = timestamp

        # управление состоянием события
        if event_recording:
            if is_exceed:
                # пока шум выше порога — сбрасываем счётчик "после"
                event_post_left = POST_EVENT_SEC
            else:
                event_post_left -= 1
                if event_post_left <= 0:
                    stop_noise_event(now)
        else:
            # не записывали — и вдруг превышение
            if is_exceed:
                start_noise_event(now, threshold)
                if event_writer is not None:
                    event_writer.write(mono)
                event_max_leq = leq_1s if leq_1s is not None else 0.0
                event_max_ts = timestamp

    except Exception as e:
        print(f"[ERROR] UMIK callback crashed: {e}")


def start_web():
    if not os.path.exists("templates/table.html"):
        print("[WARNING] Шаблон templates/table.html не найден!")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    init_csv()
    init_db()

    # запуск репортера 10-минутных JSON-ов
    start_reporter()

    # веб-интерфейс
    threading.Thread(target=start_web, daemon=True).start()

    # загрузчик событий (JSON + audio)
    threading.Thread(target=uploader_worker, daemon=True).start()

    # основной поток: UMIK-1
    with sd.InputStream(
        device=get_umick_index(),
        channels=1,
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLE_RATE * DURATION,
    ):
        print("🎤 Запись с UMIK-1... (нажмите Ctrl+C для выхода)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nЗавершено.")
