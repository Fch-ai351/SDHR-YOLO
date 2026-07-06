#!/bin/bash
# Train YOLOv8-S baseline on VisDrone2019-DET

PYTHONPATH=/path/to/FCM-main yolo detect train \
  model=/path/to/SDHR-YOLO/configs/yolov8s_baseline.yaml \
  data=/path/to/SDHR-YOLO/configs/visdrone.yaml \
  imgsz=640 \
  epochs=150 \
  batch=16 \
  workers=1 \
  close_mosaic=10 \
  seed=0 \
  deterministic=True \
  project=runs/visdrone \
  name=yolov8s_e150
