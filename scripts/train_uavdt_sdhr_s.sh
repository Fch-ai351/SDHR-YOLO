#!/usr/bin/env bash
set -euo pipefail

yolo detect train \
  model=configs/yolov8s_sdhr_3head.yaml \
  data=configs/uavdt.yaml \
  epochs=200 \
  batch=12 \
  imgsz=640 \
  optimizer=auto \
  seed=0 \
  deterministic=True \
  close_mosaic=10 \
  fliplr=0.5 \
  amp=True \
  project=runs/uavdt \
  name=sdhr_yolo_s_e200_seed0
