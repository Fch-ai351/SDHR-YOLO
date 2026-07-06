# SDHR-YOLO

This repository provides the configuration files, modified model definition files, training scripts, validation commands, inference commands, and environment information for the paper:

**SDHR-YOLO: Shallow Detail and High-Frequency Residual Learning for UAV Small Object Detection**

## Overview

SDHR-YOLO is a lightweight UAV small-object detector based on the Ultralytics YOLOv8 framework. It introduces shallow detail perception and high-frequency residual learning to enhance fine-grained spatial details and frequency-domain structural cues for UAV small-object detection.

The proposed model includes:

- Shallow Detail Perception Module (SDPM)
- High-Frequency Residual Module (HFRM)
- Shallow Detail and High-Frequency Residual (SDHR) block

## Repository Structure

```text
SDHR-YOLO/
├── configs/
│   ├── yolov8s_sdhr.yaml
│   ├── yolov8s_baseline.yaml
│   ├── visdrone.yaml
│   └── uavdt.yaml
├── sdhr_modules/
│   ├── conv.py
│   ├── __init__.py
│   ├── tasks.py
│   └── README.md
├── scripts/
│   ├── train_visdrone_sdhr_s.sh
│   ├── train_visdrone_yolov8s.sh
│   ├── train_uavdt_sdhr_s.sh
│   ├── train_uavdt_yolov8s.sh
│   ├── val.sh
│   └── infer.sh
├── docs/
│   └── environment.md
├── requirements.txt
├── LICENSE
└── README.md
