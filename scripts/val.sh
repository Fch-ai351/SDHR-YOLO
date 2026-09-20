#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/val.sh WEIGHTS [DATA_YAML]"
  echo "Example:"
  echo "  bash scripts/val.sh runs/visdrone/sdhr_yolo_s_e150_seed0/weights/best.pt configs/visdrone.yaml"
  exit 1
fi

WEIGHTS="$1"
DATA="${2:-configs/visdrone.yaml}"

yolo detect val \
  model="$WEIGHTS" \
  data="$DATA" \
  imgsz=640
