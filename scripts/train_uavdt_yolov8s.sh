#!/bin/bash
# Train YOLOv8-S baseline on UAVDT-YOLO external validation dataset

PYTHONPATH=/path/to/FCM-main yolo detect train \
  model=/path/to/SDHR-YOLO/configs/yolov8s_baseline.yaml \
  data=/path/to/SDHR-YOLO/configs/uavdt.yaml \
  imgsz=640 \
  epochs=200 \
  batch=16 \
  workers=1 \
  close_mosaic=10 \
  seed=0 \
  deterministic=True \
  patience=0 \
  project=runs/uavdt \
  name=yolov8s_uavdt_e200_noearlystop
