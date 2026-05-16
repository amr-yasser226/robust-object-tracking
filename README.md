# Bowling Score Detection from Video

![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)

DSAI 352 — Computer Vision Final Project, Spring 2026.
**Authors:** Amr Yasser · Omar Hazem

---

## What This Does

Takes a recorded bowling video as input and produces:
- An **annotated output video** with a green circle drawn on each pin from the moment it falls, labelled with the fall timestamp (MM:SS)
- A **corner HUD** on every frame showing elapsed time and current pin-down count
- A **final summary overlay** for the last 2 seconds: total time and total pins knocked down
- An **`events_log.json`** with the exact frame and timestamp of every fall event

The detection model is a YOLOv8s fine-tuned on a custom bowling dataset (mAP@50 = 98.19%). Tracking uses ByteTrack (built into Ultralytics). Fall detection is a 3-state machine (STANDING → MAYBE_FALLING → FALLEN) with aspect-ratio gating, centroid-drop gating, and ball-proximity gating to suppress false positives.

---

## Repository Structure

```
├── data/
│   └── dataset_clean/          # Curated dataset (407 train / 100 val / 50 test)
├── docs/                       # Project specification and implementation plan (PDF)
├── inference/
│   ├── inputs/                 # ← PUT YOUR VIDEO HERE
│   ├── outputs/                # ← ANNOTATED VIDEO APPEARS HERE
│   ├── main.py                 # Entry point — runs the full pipeline
│   └── src/
│       ├── fall_detector.py    # 3-state machine
│       └── renderer.py         # OpenCV drawing helpers
├── models/
│   ├── pretrained/             # yolov8s.pt, yolo11n.pt
│   └── trained/                # best.pt, last.pt  ← trained weights
├── notebooks/
│   ├── extract_frames.ipynb    # Frame extraction from raw videos
│   └── main_train.ipynb        # YOLOv8 training on Kaggle
├── reports/
│   ├── report.pdf              # Technical report
│   └── presentation.pdf        # Beamer presentation
├── results/
│   ├── training_logs/          # results.csv, metrics.json, args.yaml, TensorBoard
│   └── sample_predictions/     # 8 sample prediction images from validation set
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your video

Copy your `.mp4` or `.mov` bowling video into:

```
inference/inputs/
```

### 3. Run the pipeline

```bash
cd inference
python main.py --input inputs/your_video.mp4
```

The annotated video is written to `inference/outputs/annotated.mp4`.
The events log is written to `inference/outputs/events_log.json`.

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | `../models/trained/best.pt` | Path to trained weights |
| `--output` | `outputs/annotated.mp4` | Output video path |
| `--conf` | `0.1` | Detection confidence threshold |
| `--imgsz` | `640` | Inference image size |

---

## Model Performance

| Metric | Validation | Test |
|--------|-----------|------|
| mAP@50 | **98.19%** | 97.19% |
| mAP@50-95 | 93.83% | 91.21% |
| Precision | 92.97% | 97.69% |
| Recall | 96.64% | 88.59% |

Best F1 confidence threshold: **0.1** (determined by sweep on validation set).

Per-class: Pin AP = 0.994 · Ball AP = 0.970

Training: YOLOv8s, 100 epochs, imgsz=640, batch=16, cosine LR, Kaggle T4 GPU.

---

## Training

To re-run training from scratch, open and execute `notebooks/main_train.ipynb` on Kaggle with the dataset attached. The notebook handles dataset cleaning, training, evaluation, and artifact export.

---

## License

MIT License — see [LICENSE](LICENSE).
