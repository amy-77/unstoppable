#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

mkdir -p "$HOME/.openclaw"
mkdir -p "$HOME/.openclaw/workspace"

"$ROOT_DIR/scripts/sync-workspace.sh"

if [ ! -f "$HOME/.openclaw/openclaw.json" ]; then
  cp "$ROOT_DIR/config/openclaw.example.json" "$HOME/.openclaw/openclaw.json"
  echo "Created ~/.openclaw/openclaw.json from config/openclaw.example.json"
else
  echo "~/.openclaw/openclaw.json already exists; left unchanged"
fi

echo "OpenClaw framework install complete"

