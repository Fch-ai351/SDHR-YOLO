# SDHR-YOLO

This repository provides the configuration files, modified model definition files, training scripts, validation commands, inference commands, and environment information for the paper:

**SDHR-YOLO: Shallow Detail and Haar-Frequency Residual Learning for UAV Small Object Detection**

## Overview

SDHR-YOLO is a lightweight UAV small-object detector based on the Ultralytics YOLOv8 framework. It combines shallow detail perception with Haar-frequency residual learning to enhance UAV small-object representation while maintaining a lightweight model structure.

The proposed model includes:

- Shallow Detail Perception Module (SDPM)
- Haar-Frequency Residual Module (HFRM)
- Shallow Detail and Haar-Frequency Residual (SDHR) block

## Repository Structure

```text
SDHR-YOLO/
├── configs/
│   ├── yolov8s_sdhr_3head.yaml
│   ├── yolov8s_baseline.yaml
│   ├── visdrone.yaml
│   ├── uavdt.yaml
│   └── south_africa_wildlife.yaml
├── sdhr_modules/
│   ├── conv.py
│   ├── wavelet.py
│   ├── __init__.py
│   ├── tasks.py
│   └── README.md
├── scripts/
│   ├── train_visdrone_sdhr_s.sh
│   ├── train_visdrone_yolov8s.sh
│   ├── train_uavdt_sdhr_s.sh
│   ├── train_uavdt_yolov8s.sh
│   ├── train_south_africa_sdhr_s.sh
│   ├── train_south_africa_yolov8s.sh
│   ├── val.sh
│   └── infer.sh
├── docs/
│   └── environment.md
├── requirements.txt
└── README.md

## Implementation Note

HFRM applies fixed Haar kernels through grouped convolution. The transformed output is divided into four equal channel blocks; the first block is omitted and the remaining three blocks are retained. Because grouped-convolution outputs are arranged per input channel, the retained representation contains mixed low- and high-frequency Haar responses rather than isolated LH, HL, and HH sub-bands. Therefore, HFRM is described as a Haar-Frequency Residual Module rather than a strictly high-frequency-only branch.

## Experimental Settings

- VisDrone2019-DET: 150 epochs, batch size 8, image size 640, seed 0
- UAVDT: 200 epochs, batch size 12, image size 640, seed 0
- South African UAV wildlife: 100 epochs, batch size 16, image size 640, seed 0

## Training

VisDrone: `bash scripts/train_visdrone_sdhr_s.sh`

UAVDT: `bash scripts/train_uavdt_sdhr_s.sh`

South African wildlife: `bash scripts/train_south_africa_sdhr_s.sh`
