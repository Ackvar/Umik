# main.py
from __future__ import annotations

import os
import json
import time
import threading
from datetime import datetime, time as dtime
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.fft import rfft, rfftfreq

import web_app  # наш файл с Flask и 10-минутными репортами
from state import set_fft
from calibration_utils import load_calibration_curve, apply_frequency_calibration
from calibration import apply_calibration
from weighting import apply_a_weighting
from utils import apply_ema
from octave_analysis import octave_band_levels
from logger import init_csv
from logger_sqlite import init_db, log_to_db, insert_event_start, update_event_end
from spl_utils import compute_spl, compute_leq

# === Основные параметры ===
DURATION = 1          # секунда на блок
SAMPLE_RATE = 48000
REFERENCE_PRESSURE = 20e-6

with open('umik_config.json') as f:
    config = json.load(f)

sensitivity = float(config.get('sensitivity', 0.0045))
WEIGHTING_MODE = config.get('weighting_mode', 'Slow')

# Калибровка
freqs, gains = load_calibration_curve("7142078_90deg.txt")

# буфер на 60 секунд для Leq_60s
leq_buffer = deque(maxlen=60)

# Пороги по ТЗ
DAY_THRESHOLD = 55.0    # 07:00–23:00
NIGHT_THRESHOLD = 45.0  # 23:00–07:00

# Параметры событий
PRE_EVENT_SEC = 15
POST_EVENT_SEC = 15
AUDIO_EVENTS_DIR = Path("public/events")
AUDIO_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

pre_event_buffer = deque(maxlen=PRE_EVENT_SEC)  # храним по 1 секунде

event_recording = False
event_post_left = 0
event_writer: sf.SoundFile | None = None
event_max_leq = 0.0
current_event_id: int | None = None
current_event_threshold: float | None = None
current_event_path: str | None = None


def get_current_threshold(now: datetime | None = None) -> float:
    if now is None:
        now = datetime.now()
    t = now.time()
    if dtime(7, 0) <= t < dtime(23, 0):
        return DAY_THRESHOLD
    else:
        return NIGHT_THRESHOLD


def get_umick_index() -> int:
    """
    Берём первое устройство с входным каналом.
    В твоём выводе sounddevice это:
    1 USB PnP Sound Device: Audio (hw:4,0), ALSA (1 in, 2 out)
    """
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            print(f"[INPUT] Беру устройство #{i}: {dev['name']}")
            return i
    raise RuntimeError("Нет ни одного входного аудио-устройства")


def start_noise_event(now: datetime, threshold: float):
    """
    Начинаем событие:
    - открываем WAV
    - дописываем буфер PRE_EVENT_SEC
    - создаём запись в БД
    """
    global event_recording, event_post_left, event_writer, event_max_leq
    global current_event_id, current_event_threshold, current_event_path

    event_recording = True
    event_post_left = POST_EVENT_SEC
    event_max_leq = 0.0
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
        subtype="PCM_16"
    )

    # дописываем историю до события
    for chunk in pre_event_buffer:
        event_writer.write(chunk)

    current_event_id = insert_event_start(ts_str, threshold, str(filepath))
    print(f"[EVENT] START id={current_event_id} file={filepath} thr={threshold:.1f} dBA")


def stop_noise_event(now: datetime):
    """
    Останавливаем событие:
    - закрываем WAV
    - считаем max_leq
    - отправляем JSON на /capture/
    - получаем id и отправляем WAV на /capture/{id}/audio/
    - обновляем запись в БД
    """
    global event_recording, event_writer, current_event_id, event_max_leq
    global current_event_threshold, current_event_path

    if not event_recording:
        return

    event_recording = False
    end_ts = now.strftime("%Y-%m-%d %H:%M:%S")

    if event_writer is not None:
        event_writer.close()
        event_writer = None

    measurement_id = None

    # Сначала пытаемся отправить JSON + аудио
    if web_app.REPORT_API_URL and current_event_path is not None:
        try:
            payload = [{
                "device_serial": web_app.DEVICE_ID,
                "value": float(event_max_leq),
                "event_time": end_ts
            }]
            print(f"[EVENT] POST JSON for event_id={current_event_id}: {payload}")

            resp = web_app._http.post(web_app.REPORT_API_URL, json=payload, timeout=10)
            if 200 <= resp.status_code < 300:
                data = resp.json()
                if isinstance(data, list) and data:
                    measurement_id = data[0].get("id")
                print(f"[EVENT] JSON OK, measurement_id={measurement_id}")
            else:
                print(f"[EVENT] JSON FAIL {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[EVENT] JSON ERROR: {e}")

        # Отправляем аудио, если есть measurement_id
        if measurement_id is not None:
            upload_url = f"{web_app.REPORT_API_URL}{measurement_id}/audio/"
            try:
                with open(current_event_path, "rb") as f:
                    files = {"audio": ("event.wav", f, "audio/wav")}
                    r2 = web_app._http.post(upload_url, files=files, timeout=30)
                if 200 <= r2.status_code < 300:
                    print(f"[EVENT] AUDIO OK -> {upload_url}")
                else:
                    print(f"[EVENT] AUDIO FAIL {r2.status_code}: {r2.text}")
            except Exception as e:
                print(f"[EVENT] AUDIO ERROR: {e}")
        else:
            print("[EVENT] measurement_id не получен, аудио не отправляем")

    # Обновляем БД
    if current_event_id is not None:
        update_event_end(current_event_id, end_ts, event_max_leq, measurement_id)
        print(
            f"[EVENT] STOP id={current_event_id} max_leq={event_max_leq:.1f} "
            f"thr={current_event_threshold:.1f} at {end_ts} meas_id={measurement_id}"
        )

    current_event_id = None
    current_event_threshold = None
    current_event_path = None


def audio_callback(indata, frames, time_info, status):
    """
    Основной callback: считает уровни, пишет в БД, управляет событиями.
    """
    global event_recording, event_post_left, event_max_leq, event_writer

    try:
        if status:
            print(f"[UMIK] Status: {status}")

        mono = indata[:, 0].astype(np.float64)
        now = datetime.now()

        # ---- форс-обрезка события по границе 10-минутного отчёта ----
        if getattr(web_app, "force_cut_event", False):
            web_app.force_cut_event = False
            if event_recording:
                print("[EVENT] Force cut by 10-min report boundary")
                stop_noise_event(now)
                pre_event_buffer.clear()
        # --------------------------------------------------------------

        # буфер для PRE_EVENT_SEC
        pre_event_buffer.append(mono.copy())
        threshold = get_current_threshold(now)

        pressure_signal = apply_calibration(mono, sensitivity)
        pressure_signal = apply_frequency_calibration(pressure_signal, SAMPLE_RATE, freqs, gains)

        # FFT для веба
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

        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        log_to_db(timestamp, spl, leq_1s, leq_60s, lmax, bands)

        # проверяем превышение
        is_exceed = (leq_1s is not None) and (leq_1s > threshold)

        # если уже пишем событие
        if event_recording and event_writer is not None:
            event_writer.write(mono)
            if leq_1s is not None and leq_1s > event_max_leq:
                event_max_leq = leq_1s

        if event_recording:
            if is_exceed:
                # шум всё ещё выше порога — продлеваем "хвост"
                event_post_left = POST_EVENT_SEC
            else:
                event_post_left -= 1
                if event_post_left <= 0:
                    stop_noise_event(now)
        else:
            # не писали — но произошло превышение
            if is_exceed:
                start_noise_event(now, threshold)
                if event_writer is not None:
                    event_writer.write(mono)
                event_max_leq = leq_1s if leq_1s is not None else 0.0

    except Exception as e:
        print(f"[ERROR] UMIK callback crashed: {e}")


def start_web():
    if not os.path.exists("templates/table.html"):
        print("[WARNING] Шаблон templates/table.html не найден!")

    # стартуем 10-минутный репортёр
    web_app.start_reporter()

    web_app.app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    init_csv()
    init_db()

    # веб-интерфейс + репортёр в отдельном потоке
    threading.Thread(target=start_web, daemon=True).start()

    # основной поток: запись с микрофона
    with sd.InputStream(
        device=get_umick_index(),
        channels=1,
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLE_RATE * DURATION
    ):
        print("🎤 Запись с микрофона... (нажмите Ctrl+C для выхода)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nЗавершено.")
