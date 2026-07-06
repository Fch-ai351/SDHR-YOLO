#!/bin/bash
# Example inference command

PYTHONPATH=/path/to/FCM-main yolo detect predict \
  model=/path/to/weights/hongwai_best.pt \
  source=/path/to/images \
  imgsz=640 \
  conf=0.25 \
  save=True
