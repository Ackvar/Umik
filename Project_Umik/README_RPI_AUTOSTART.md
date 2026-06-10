# Raspberry Pi autostart

This project can be started on boot with a `systemd` service.

## Install

Copy the project to Raspberry Pi, open a terminal in the project folder, then run:

```bash
chmod +x install_rpi_autostart.sh
sudo ./install_rpi_autostart.sh
```

By default the service starts `main.py`. That script also starts the Flask web UI on port `5000`.
The installer also installs typical Raspberry Pi packages: `python3-venv`, `python3-pip`, `portaudio19-dev`, `libsndfile1`, and `ffmpeg`.
On every service start, `start_rpi_app.sh` checks the virtual environment and installs missing Python packages from `requirements.txt`, then runs `main.py`.

To start the full app manually with the same checks:

```bash
chmod +x start_rpi_app.sh
./start_rpi_app.sh
```

To start only the web UI:

```bash
sudo ./install_rpi_autostart.sh --web-only
```

To skip `apt` package installation:

```bash
sudo SKIP_APT=1 ./install_rpi_autostart.sh
```

## Service commands

```bash
sudo systemctl status umik
sudo journalctl -u umik -f
sudo systemctl restart umik
sudo systemctl disable --now umik
```

## Settings

Optional environment variables can be edited here:

```bash
sudo nano /etc/default/umik
sudo systemctl restart umik
```

Useful variables:

```bash
EVENT_THRESHOLD_DB=45
REPORT_API_URL=https://shum.i20h.ru/api/v1/measurements/capture
NOISE_RAW_ENABLED=0
UMIK_DEVICE_INDEX=0
```
