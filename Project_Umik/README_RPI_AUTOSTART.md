# Raspberry Pi autostart

This project can be started on boot with a `systemd` service.

## One Command Setup

After downloading/copying the project to Raspberry Pi, run this once from the project folder:

```bash
bash setup_rpi.sh
```

This installs dependencies, creates the Python environment, enables autostart, and starts the app through `main.py`.

If you want the app to open in a visible terminal window after desktop login:

```bash
bash setup_rpi.sh terminal
```

## Install

Copy the project to Raspberry Pi, open a terminal in the project folder, then run:

```bash
chmod +x install_rpi_autostart.sh
sudo ./install_rpi_autostart.sh
```

If the file was copied from Windows and does not start, run it through bash:

```bash
sudo bash install_rpi_autostart.sh
```

By default the service starts `main.py`. That script also starts the Flask web UI on port `5000`.
The installer also installs typical Raspberry Pi packages: `python3-venv`, `python3-pip`, `portaudio19-dev`, `libsndfile1`, and `ffmpeg`.
On every service start, `start_rpi_app.sh` checks the virtual environment and installs missing Python packages from `requirements.txt`, then runs `main.py`.

To start the full app manually with the same checks:

```bash
chmod +x start_rpi_app.sh
./start_rpi_app.sh
```

## Open In Terminal On Desktop Login

If Raspberry Pi boots into the graphical desktop and you want to see the command window, install desktop autostart:

```bash
sudo systemctl disable --now umik
chmod +x install_rpi_terminal_autostart.sh
./install_rpi_terminal_autostart.sh
```

This creates:

```bash
~/.config/autostart/umik-terminal.desktop
```

It opens a terminal window through `open_rpi_terminal_app.sh` and runs `start_rpi_app.sh`, which checks Python/packages and then starts `main.py`.

To disable this visible terminal autostart:

```bash
rm ~/.config/autostart/umik-terminal.desktop
```

This desktop mode starts after the user logs into the graphical Raspberry Pi desktop. For startup before login, use the `systemd` service.

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

## If autostart does not work

Run diagnostics:

```bash
chmod +x diagnose_rpi_autostart.sh
./diagnose_rpi_autostart.sh
```

The most important checks are:

```bash
systemctl is-enabled umik
sudo systemctl status umik
sudo journalctl -u umik -n 120 --no-pager
```

Reinstall the service after copying updated scripts:

```bash
sudo bash install_rpi_autostart.sh
sudo systemctl restart umik
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
