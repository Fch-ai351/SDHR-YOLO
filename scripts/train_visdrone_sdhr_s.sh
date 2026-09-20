#!/usr/bin/env bash
set -euo pipefail

yolo detect train \
  model=configs/yolov8s_sdhr_3head.yaml \
  data=configs/visdrone.yaml \
  epochs=150 \
  batch=8 \
  imgsz=640 \
  optimizer=auto \
  seed=0 \
  deterministic=True \
  close_mosaic=10 \
  fliplr=0.5 \
  amp=True \
  project=runs/visdrone \
  name=sdhr_yolo_s_e150_seed0
