#!/usr/bin/env bash
set -euo pipefail

yolo detect train \
  model=configs/yolov8s_baseline.yaml \
  data=configs/south_africa_wildlife.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  optimizer=auto \
  seed=0 \
  deterministic=True \
  close_mosaic=10 \
  fliplr=0.5 \
  amp=True \
  project=runs/south_africa \
  name=yolov8s_e100_seed0
