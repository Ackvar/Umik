#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-UMIK Monitor}"
APP_SCRIPT="${APP_SCRIPT:-main.py}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_CMD="cd \"${PROJECT_DIR}\" && APP_SCRIPT=\"${APP_SCRIPT}\" ./start_rpi_app.sh; echo; echo Press Enter to close; read"

if command -v lxterminal >/dev/null 2>&1; then
  exec lxterminal --title="${APP_NAME}" --command="bash -lc '${RUN_CMD}'"
elif command -v x-terminal-emulator >/dev/null 2>&1; then
  exec x-terminal-emulator -T "${APP_NAME}" -e bash -lc "${RUN_CMD}"
elif command -v gnome-terminal >/dev/null 2>&1; then
  exec gnome-terminal --title="${APP_NAME}" -- bash -lc "${RUN_CMD}"
else
  echo "No supported terminal found. Install one, for example: sudo apt install lxterminal" >&2
  exit 1
fi
