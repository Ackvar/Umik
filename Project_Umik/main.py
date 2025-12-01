import os
import json
import time
import sqlite3
import threading
from datetime import datetime
from collections import deque
from pathlib import Path
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.fft import rfft, rfftfreq

from web_app import app
from state import set_fft
from calibration_utils import load_calibration_curve, apply_frequency_calibration
from calibration import apply_calibration
from weighting import apply_a_weighting
from utils import apply_ema
from octave_analysis import octave_band_levels
from logger import init_csv, log_to_csv
from logger_sqlite import init_db, log_to_db
from spl_utils import compute_spl, compute_leq

# === Конфигурация (UMIK остаётся как было) ===
DURATION = 1
SAMPLE_RATE = 48000
REFERENCE_PRESSURE = 20e-6

with open('umik_config.json') as f:
    config = json.load(f)

sensitivity = float(config.get('sensitivity', 0.0045))
WEIGHTING_MODE = config.get('weighting_mode', 'Slow')

# === Настройки USB/analog (второй микрофон) ===
ANALOG_DEVICE_SUBSTR = os.getenv('ANALOG_DEVICE_SUBSTR', config.get('analog_device_substr', ''))  # напр. "USB"
ANALOG_SR        = int(os.getenv('ANALOG_SR',        config.get('analog_sr', SAMPLE_RATE)))
ANALOG_CH        = int(os.getenv('ANALOG_CHANNELS',  config.get('analog_channels', 1)))
ANALOG_BLOCK     = int(os.getenv('ANALOG_BLOCK',     config.get('analog_block', 2048)))
SEGMENT_SEC      = int(os.getenv('ANALOG_SEGMENT_SEC', config.get('analog_segment_sec', 300)))  # 5 минут
OUT_PATH         = Path(os.getenv('ANALOG_OUT_PATH', config.get('analog_out_path', 'public/analog.wav')))

# === Калибровка для UMIK ===
freqs, gains = load_calibration_curve("7142078_90deg.txt")

# === Буфер Leq для UMIK (60 последних секунд) ===
leq_buffer = deque(maxlen=60)

# === БД (как было) ===
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

# === Поиск устройств ===
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
        if dev['max_input_channels'] > 0 and "umik-1" in dev['name'].lower():
            return i
    raise RuntimeError("UMIK-1 не найден")

# === UMIK поток (НЕ МЕНЯЛСЯ по сути) ===
def audio_callback(indata, frames, time_info, status):
    try:
        if status:
            print(f"[UMIK] Status: {status}")

        mono = indata[:, 0].astype(np.float64)
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

        print(f"SPL: {spl:.1f} dBA | Leq_1s: {leq_1s:.1f} dBA | Leq_60s: {leq_60s:.1f} dBA | Lmax: {lmax:.1f} dBA")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_to_db(timestamp, spl, leq_1s, leq_60s, lmax, bands)

    except Exception as e:
        print(f"[ERROR] UMIK callback crashed: {e}")

# === USB/Analog запись 5-минутными файлами (перезапись) ===
def start_analog_recorder():
    """Запись с USB микрофона в кольцевом режиме: каждые SEGMENT_SEC секунд перезаписываем WAV."""
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
            # кладём копию блока в очередь для писателя
            q.put(indata.copy())

        print(f"[ANALOG] Используется устройство: {sd.query_devices()[idx]['name']} (idx={idx})")
        with sd.InputStream(device=idx, channels=ANALOG_CH, samplerate=ANALOG_SR,
                            blocksize=ANALOG_BLOCK, callback=analog_cb, dtype='float32'):
            print("🎙️ Аналоговый микрофон: запись по 5 минут с перезаписью файла.")
            while True:
                # открываем/перезаписываем файл на следующий сегмент
                with sf.SoundFile(str(OUT_PATH), mode='w', samplerate=ANALOG_SR,
                                  channels=ANALOG_CH, subtype='PCM_16') as wav:
                    t_end = time.time() + SEGMENT_SEC
                    while time.time() < t_end:
                        try:
                            block = q.get(timeout=0.5)
                            wav.write(block)
                        except Empty:
                            pass
                # после 5 минут файл просто закрывается; цикл начнётся заново и файл будет перезаписан
                print(f"[ANALOG] Сегмент {SEGMENT_SEC}s записан → {OUT_PATH.name} (перезапись началась заново)")

    except Exception as e:
        print(f"[ERROR] Аналоговый рекордер не запустился: {e}")

# === Веб-сервер ===
def start_web():
    if not os.path.exists("templates/table.html"):
        print("[WARNING] Шаблон templates/table.html не найден!")
    app.run(host="0.0.0.0", port=5000, debug=False)

# === Главная точка входа ===
if __name__ == "__main__":
    init_csv()
    init_db()
    init_weighted_table()

    # веб
    threading.Thread(target=start_web, daemon=True).start()

    # параллельная запись с USB/analog (только если указан в конфиге)
    if ANALOG_DEVICE_SUBSTR:
        threading.Thread(target=start_analog_recorder, daemon=True).start()
    else:
        print("[ANALOG] analog_device_substr пуст — аналоговая запись отключена.")

    # основной поток: UMIK-1 (как раньше)
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
