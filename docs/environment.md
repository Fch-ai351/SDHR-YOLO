# Experimental environment

The experiments in the manuscript were conducted under the following environment:

- OS: Ubuntu 22.04
- CPU: Intel Core i5-14600KF
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB
- Python: 3.10.19
- PyTorch: 2.7.0 + CUDA 12.8
- Ultralytics YOLO: 8.0.137
- Input image size: 640 x 640
- AMP: enabled

Main random seeds:
- Main comparison: seed 0
- Random seed stability analysis: seeds 0, 1, and 2

Main training schedules:
- VisDrone2019-DET main comparison: 150 epochs
- UAVDT-YOLO external validation: 200 epochs, patience=0
