#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/infer.sh WEIGHTS SOURCE"
  echo "Example:"
  echo "  bash scripts/infer.sh path/to/best.pt path/to/images"
  exit 1
fi

WEIGHTS="$1"
SOURCE="$2"

yolo detect predict \
  model="$WEIGHTS" \
  source="$SOURCE" \
  imgsz=640 \
  conf=0.25 \
  save=True
