#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-umik}"
APP_SCRIPT="${APP_SCRIPT:-main.py}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_USER="${SUDO_USER:-$(id -un)}"
VENV_DIR="${PROJECT_DIR}/.runtime-venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
START_SCRIPT="${PROJECT_DIR}/start_rpi_app.sh"
DIAG_SCRIPT="${PROJECT_DIR}/diagnose_rpi_autostart.sh"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_FILE="/etc/default/${SERVICE_NAME}"

usage() {
  cat <<USAGE
Usage:
  sudo ./install_rpi_autostart.sh [--web-only] [--script FILE.py]

Options:
  --web-only        Run only web_app.py instead of the full measurement app.
  --script FILE.py  Run a custom Python entrypoint from this project.

Environment overrides:
  sudo SERVICE_NAME=umik APP_SCRIPT=main.py SKIP_APT=1 ./install_rpi_autostart.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --web-only)
      APP_SCRIPT="web_app.py"
      shift
      ;;
    --script)
      APP_SCRIPT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this script with sudo: sudo ./install_rpi_autostart.sh" >&2
  exit 1
fi

if [[ -z "${APP_SCRIPT}" || ! -f "${PROJECT_DIR}/${APP_SCRIPT}" ]]; then
  echo "App script not found: ${PROJECT_DIR}/${APP_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${START_SCRIPT}" ]]; then
  echo "Start script not found: ${START_SCRIPT}" >&2
  exit 1
fi

if [[ -f "${DIAG_SCRIPT}" ]]; then
  sed -i 's/\r$//' "${DIAG_SCRIPT}"
  chmod +x "${DIAG_SCRIPT}"
fi

echo "[1/6] Installing Raspberry Pi system packages"
if [[ "${SKIP_APT:-0}" == "1" ]]; then
  echo "SKIP_APT=1, skipping apt packages"
elif command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y python3-venv python3-pip portaudio19-dev libsndfile1 ffmpeg
else
  echo "apt-get not found, skipping system packages"
fi

echo "[2/6] Preparing Python virtual environment"
sed -i 's/\r$//' "${START_SCRIPT}"
chmod +x "${START_SCRIPT}"
sudo -u "${SERVICE_USER}" python3 -m venv "${VENV_DIR}"
sudo -u "${SERVICE_USER}" "${PYTHON_BIN}" -m pip install --upgrade pip
sudo -u "${SERVICE_USER}" "${PYTHON_BIN}" -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo "[3/6] Creating environment file: ${ENV_FILE}"
if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<'ENV'
# Optional app settings for the UMIK service.
# Uncomment and edit values if needed.
# EVENT_THRESHOLD_DB=45
# REPORT_API_URL=https://shum.i20h.ru/api/v1/measurements/capture
# NOISE_RAW_ENABLED=0
# UMIK_DEVICE_INDEX=0
ENV
fi

echo "[4/6] Installing systemd service: ${SERVICE_FILE}"
cat > "${SERVICE_FILE}" <<SERVICE
[Unit]
Description=UMIK noise monitoring app
Wants=network-online.target
After=network-online.target sound.target
ConditionPathExists=${PROJECT_DIR}/${APP_SCRIPT}
ConditionPathExists=${START_SCRIPT}

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=APP_SCRIPT=${APP_SCRIPT}
EnvironmentFile=-${ENV_FILE}
ExecStart=/bin/bash ${START_SCRIPT}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE

echo "[5/6] Enabling and starting ${SERVICE_NAME}.service"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
systemctl restart "${SERVICE_NAME}.service"

echo "[6/6] Done"
echo
echo "Status:"
systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
echo
echo "Useful commands:"
echo "  sudo systemctl status ${SERVICE_NAME}"
echo "  sudo journalctl -u ${SERVICE_NAME} -f"
echo "  sudo systemctl restart ${SERVICE_NAME}"
echo "  sudo systemctl disable --now ${SERVICE_NAME}"
