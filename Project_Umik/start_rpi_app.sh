#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_SCRIPT="${APP_SCRIPT:-main.py}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.runtime-venv}"
PYTHON3="${PYTHON3:-python3}"
REQUIREMENTS_FILE="${PROJECT_DIR}/requirements.txt"
PYTHON_BIN="${VENV_DIR}/bin/python"
STAMP_FILE="${VENV_DIR}/.requirements.stamp"

log() {
  echo "[UMIK-START] $*"
}

cd "${PROJECT_DIR}"

if [[ ! -f "${PROJECT_DIR}/${APP_SCRIPT}" ]]; then
  echo "[UMIK-START] App script not found: ${PROJECT_DIR}/${APP_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "[UMIK-START] requirements.txt not found: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

if ! command -v "${PYTHON3}" >/dev/null 2>&1; then
  echo "[UMIK-START] python3 is not installed. Install it with: sudo apt install python3 python3-venv python3-pip" >&2
  exit 127
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  log "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON3}" -m venv "${VENV_DIR}"
fi

log "Checking pip"
"${PYTHON_BIN}" -m ensurepip --upgrade >/dev/null 2>&1 || true

needs_install=0
if [[ ! -f "${STAMP_FILE}" || "${REQUIREMENTS_FILE}" -nt "${STAMP_FILE}" ]]; then
  needs_install=1
else
  if ! "${PYTHON_BIN}" - "${REQUIREMENTS_FILE}" <<'PY'
import importlib.metadata as metadata
import re
import sys

requirements_path = sys.argv[1]
missing = []

with open(requirements_path, "r", encoding="utf-8") as requirements:
    for raw_line in requirements:
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        package = re.split(r"[<>=!~;\s\[]", line, maxsplit=1)[0].strip()
        if not package:
            continue
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package)

if missing:
    print("Missing packages: " + ", ".join(missing))
    sys.exit(1)
PY
  then
    needs_install=1
  fi
fi

if [[ "${needs_install}" == "1" ]]; then
  log "Installing Python dependencies from requirements.txt"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r "${REQUIREMENTS_FILE}"
  touch "${STAMP_FILE}"
else
  log "Python dependencies are ready"
fi

log "Starting ${APP_SCRIPT}"
exec "${PYTHON_BIN}" "${PROJECT_DIR}/${APP_SCRIPT}"
