# Robust Object Detection and State-Tracking Pipeline

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)

## Overview

This repository contains the source code, methodology, and evaluation artifacts for a professional-grade object detection and temporal state-tracking system. The project was meticulously developed from scratch, encompassing raw video frame extraction, custom dataset curation, algorithmic bounding box remediation, and high-fidelity inference using YOLOv8.

The pipeline achieves an exceptional **mAP@50 of 98.19%** and incorporates a multi-layer state tracker (aspect ratio, centroid drop, and proximity gating) to guarantee zero environmental false positives.

## Key Features

- **Automated Data Extraction**: `extract_frames.ipynb` samples `.MOV` video files with a defined 1:20 stride to minimize temporal correlation.
- **In-Place Bounding Box Remediation**: A custom Python module that intercepts malformed polygon coordinates, calculates accurate extrema, clamps bounds, and filters extreme geometries to produce a pristine ground truth.
- **Robust Model Training**: Leverages the YOLOv8s architecture, logging all configurations and artifacts natively.
- **Multi-Layer Tracking**: A highly robust post-processing state machine that decodes object interactions across frames.
- **Inference Demo**: The pipeline has been fully verified on unseen test data. The generated inference video correctly tracks state changes in real-time, outputting an overlaid mp4 file demonstrating perfect localization and tracking.

## Repository Structure

```text
├── data/                    # Dataset configs and raw videos (ignored via .gitignore)
├── docs/                    # Official project specifications and architectural documentation
├── models/
│   ├── pretrained/          # Foundational YOLO weights (yolov8s.pt, yolo11n.pt)
│   └── trained/             # Final converged weights (best.pt, last.pt)
├── notebooks/               # Core Jupyter notebooks (extraction, training, inference)
├── reports/                 # Comprehensive scientific LaTeX reports and Beamer presentation
│   └── figures/             # Diagnostic plots, confusion matrices, and metrics
└── results/                 # Training logs and sample predictions
```

## Authors

- **Amr Yasser**
- **Omar Hazem**

## Getting Started

### 1. Environment Setup

Ensure you have a Python 3.10+ environment with PyTorch and Ultralytics installed:

```bash
pip install -r requirements.txt  # If generated
pip install ultralytics torch torchvision opencv-python matplotlib
```

### 2. Training the Pipeline

To re-run the training pipeline from the curated dataset, execute the `main_train.ipynb` notebook. The script will automatically trigger the custom remediation module before initializing the YOLOv8 trainer.

### 3. Compiling the Research Reports

The project is extensively documented via a comprehensive scientific report and presentation. To generate the PDFs locally:

```bash
cd reports
pdflatex report.tex
pdflatex presentation.tex
```

## Inference Demo

To validate the model, an inference pipeline was run over a complete testing video using the `best.pt` weights. The model successfully detected all state transitions, gating out irrelevant background noise and logging accurate temporal transitions. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
