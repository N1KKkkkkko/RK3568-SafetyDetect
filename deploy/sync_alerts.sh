#!/usr/bin/env bash
# Sync SafeDetect alert records from the board to a NAS/PC via rsync over SSH.
# Data stays within your LAN / Tailscale tunnel (no cloud).
# Usage:  sudo bash sync_alerts.sh
# Config via env vars:
#   ALERTS_DIR   source dir (default below)
#   REMOTE_HOST  user@host of the NAS/PC
#   REMOTE_PATH  target dir on the NAS/PC
set -euo pipefail

ALERTS_DIR="${ALERTS_DIR:-/home/firefly/rk3568_camera/SafeDetect_V0.01/alerts}/"
REMOTE_HOST="${REMOTE_HOST:-user@nas_ip}"
REMOTE_PATH="${REMOTE_PATH:-/volume1/backup/safe_alerts/}"

if [ "${REMOTE_HOST}" = "user@nas_ip" ]; then
  echo "ERROR: set REMOTE_HOST first, e.g. REMOTE_HOST=user@<板子IP>"
  exit 1
fi
if [ ! -d "${ALERTS_DIR}" ]; then
  echo "WARN: source dir missing: ${ALERTS_DIR}"
  exit 0
fi
rsync -az --delete "${ALERTS_DIR}" "${REMOTE_HOST}:${REMOTE_PATH}"
echo "alerts synced: $(date '+%F %T')"