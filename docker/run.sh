#!/usr/bin/env bash
# Usage:
#   docker/run.sh                       # interactive shell in the container
#   docker/run.sh python scripts/track_video.py --video assets/apple.mp4
#
# Env vars:
#   GPU=1          which GPU to expose (default: 0)
#   DATA=/path     extra host dir with your videos, mounted at /data (optional)
set -euo pipefail

IMAGE=cotracker3:latest
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${GPU:-0}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" -f "$REPO/docker/Dockerfile" "$REPO/docker"
fi

EXTRA=()
if [[ -n "${DATA:-}" ]]; then
    EXTRA+=(-v "$(realpath "$DATA")":/data)
fi

TTY=()
[[ -t 0 ]] && TTY=(-it)

exec docker run --rm "${TTY[@]}" \
    --gpus "device=$GPU" \
    --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$REPO":/workspace \
    "${EXTRA[@]}" \
    "$IMAGE" "${@:-bash}"
