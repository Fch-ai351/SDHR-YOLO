#!/bin/bash
# Example validation command

PYTHONPATH=/path/to/FCM-main yolo detect val \
  model=/path/to/weights/hongwai_best.pt \
  data=/path/to/SDHR-YOLO/configs/visdrone.yaml \
  imgsz=640 \
  batch=16 \
  workers=1
