#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-service}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<USAGE
Usage:
  bash setup_rpi.sh [service|terminal]

Modes:
  service   Start automatically in the background on Raspberry Pi boot.
  terminal  Open a visible terminal after graphical desktop login.

Examples:
  bash setup_rpi.sh
  bash setup_rpi.sh terminal
USAGE
}

clean_script() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    sed -i 's/\r$//' "${path}"
    chmod +x "${path}"
  fi
}

case "${MODE}" in
  service|terminal)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 2
    ;;
esac

cd "${PROJECT_DIR}"

clean_script "${PROJECT_DIR}/start_rpi_app.sh"
clean_script "${PROJECT_DIR}/install_rpi_autostart.sh"
clean_script "${PROJECT_DIR}/install_rpi_terminal_autostart.sh"
clean_script "${PROJECT_DIR}/open_rpi_terminal_app.sh"
clean_script "${PROJECT_DIR}/diagnose_rpi_autostart.sh"

if [[ "${MODE}" == "service" ]]; then
  echo "[SETUP] Installing background boot autostart"
  sudo bash "${PROJECT_DIR}/install_rpi_autostart.sh"
  echo
  echo "[SETUP] Done. The app will start automatically on boot."
  echo "[SETUP] Check it with: sudo systemctl status umik"
else
  echo "[SETUP] Installing visible terminal autostart"
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl disable --now umik 2>/dev/null || true
  fi
  bash "${PROJECT_DIR}/install_rpi_terminal_autostart.sh"
  echo
  echo "[SETUP] Done. The app will open in a terminal after desktop login."
fi
