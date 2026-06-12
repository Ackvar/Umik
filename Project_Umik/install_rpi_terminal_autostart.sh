#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-UMIK Monitor}"
APP_SCRIPT="${APP_SCRIPT:-main.py}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="${PROJECT_DIR}/start_rpi_app.sh"
OPEN_SCRIPT="${PROJECT_DIR}/open_rpi_terminal_app.sh"
AUTOSTART_DIR="${HOME}/.config/autostart"
DESKTOP_FILE="${AUTOSTART_DIR}/umik-terminal.desktop"

if [[ ! -f "${START_SCRIPT}" ]]; then
  echo "Start script not found: ${START_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/${APP_SCRIPT}" ]]; then
  echo "App script not found: ${PROJECT_DIR}/${APP_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${OPEN_SCRIPT}" ]]; then
  echo "Terminal opener script not found: ${OPEN_SCRIPT}" >&2
  exit 1
fi

sed -i 's/\r$//' "${START_SCRIPT}"
sed -i 's/\r$//' "${OPEN_SCRIPT}"
chmod +x "${START_SCRIPT}"
chmod +x "${OPEN_SCRIPT}"
mkdir -p "${AUTOSTART_DIR}"

if ! command -v lxterminal >/dev/null 2>&1 \
  && ! command -v x-terminal-emulator >/dev/null 2>&1 \
  && ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "No supported terminal found. Install one, for example: sudo apt install lxterminal" >&2
  exit 1
fi

cat > "${DESKTOP_FILE}" <<DESKTOP
[Desktop Entry]
Type=Application
Name=${APP_NAME}
Comment=Start UMIK monitor in a terminal window
Exec=env APP_SCRIPT=${APP_SCRIPT} /bin/bash "${OPEN_SCRIPT}"
Terminal=false
X-GNOME-Autostart-enabled=true
DESKTOP

chmod +x "${DESKTOP_FILE}"

echo "Desktop autostart installed:"
echo "  ${DESKTOP_FILE}"
echo
echo "It will open a terminal and run ${APP_SCRIPT} after this user logs into the Raspberry Pi desktop."
echo "If the systemd service is enabled too, disable it to avoid running the app twice:"
echo "  sudo systemctl disable --now umik"
echo "To test now, log out and back in, or run:"
echo "  ${OPEN_SCRIPT}"
