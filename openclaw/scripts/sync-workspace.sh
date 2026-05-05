#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

mkdir -p "$HOME/.openclaw/workspace"

rsync -a \
  --exclude '.DS_Store' \
  --exclude '.git' \
  "$ROOT_DIR/workspace/" "$HOME/.openclaw/workspace/"

echo "Synced workspace to ~/.openclaw/workspace"

