#!/bin/sh
set -eu
env_file="$1"
shift
if [ -f "$env_file" ]; then
  . "$env_file"
fi
exec "$@"

