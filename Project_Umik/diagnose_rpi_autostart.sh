#!/usr/bin/env bash
set -u

SERVICE_NAME="${SERVICE_NAME:-umik}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "== UMIK autostart diagnostics =="
echo "Project: ${PROJECT_DIR}"
echo "Service: ${SERVICE_NAME}"
echo

echo "== Files =="
ls -l "${PROJECT_DIR}/install_rpi_autostart.sh" 2>/dev/null || true
ls -l "${PROJECT_DIR}/start_rpi_app.sh" 2>/dev/null || true
ls -l "${PROJECT_DIR}/main.py" 2>/dev/null || true
ls -l "${PROJECT_DIR}/requirements.txt" 2>/dev/null || true
echo

echo "== Python =="
command -v python3 || true
python3 --version 2>/dev/null || true
if [[ -x "${PROJECT_DIR}/.runtime-venv/bin/python" ]]; then
  "${PROJECT_DIR}/.runtime-venv/bin/python" --version || true
else
  echo "No virtualenv python found at ${PROJECT_DIR}/.runtime-venv/bin/python"
fi
echo

echo "== systemd enabled =="
systemctl is-enabled "${SERVICE_NAME}" 2>&1 || true
echo

echo "== systemd status =="
systemctl --no-pager --full status "${SERVICE_NAME}" 2>&1 || true
echo

echo "== service file =="
if [[ -f "${SERVICE_FILE}" ]]; then
  sed -n '1,160p' "${SERVICE_FILE}"
else
  echo "Service file not found: ${SERVICE_FILE}"
fi
echo

echo "== last logs =="
journalctl -u "${SERVICE_NAME}" -n 120 --no-pager 2>&1 || true
