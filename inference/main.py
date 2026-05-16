"""
Bowling score detection — full inference pipeline.

Usage:
    python main.py --input inputs/game.mp4

Outputs (both written to the same folder as --output):
    annotated.mp4   — re-rendered video with markers, timestamps, score overlay
    events_log.json — machine-readable log of every pin-fall event
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from src.fall_detector import FallDetector
from src.renderer import draw_frame

CLASS_BALL = 0
CLASS_PIN = 1


def parse_args():
    p = argparse.ArgumentParser(description="Bowling pin fall detection pipeline.")
    p.add_argument("--input", required=True,
                   help="Path to input video (e.g. inputs/game.mp4).")
    p.add_argument("--weights", default="../models/trained/best.pt",
                   help="Path to trained best.pt weights.")
    p.add_argument("--output", default="outputs/annotated.mp4",
                   help="Path for the annotated output video.")
    p.add_argument("--conf", type=float, default=0.1,
                   help="Detection confidence threshold (default: 0.1).")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference image size (default: 640).")
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input video not found: {input_path}")
        sys.exit(1)
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        print("        Make sure you are running from inside the inference/ directory.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Video metadata ---
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[INFO] Input  : {input_path}  ({width}x{height} @ {fps:.1f} fps, {total_frames} frames)")
    print(f"[INFO] Weights: {weights_path}")
    print(f"[INFO] Output : {output_path}")

    # --- Load model (import here so startup errors are clear) ---
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(str(weights_path))

    # --- Video writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # --- Pipeline state ---
    detector = FallDetector(fps)
    fallen_pins: dict = {}   # track_id -> fall_timestamp_seconds
    events_log: list = []

    print("[INFO] Running … (press Ctrl+C to abort)\n")

    cap = cv2.VideoCapture(str(input_path))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track pins and ball across frames using ByteTrack (built into Ultralytics)
        results = model.track(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        # --- Parse detections ---
        pin_centroids: dict = {}
        boxes = results.boxes

        if boxes is not None and boxes.id is not None:
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                tid = int(boxes.id[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if cls == CLASS_BALL:
                    detector.update_ball(frame_idx, cx, cy)

                elif cls == CLASS_PIN:
                    pin_centroids[tid] = (cx, cy)
                    just_fell = detector.update_pin(tid, frame_idx, x1, y1, x2, y2)
                    if just_fell:
                        ts = detector.get_fall_timestamp(tid)
                        fallen_pins[tid] = ts
                        m = int(ts) // 60
                        s = int(ts) % 60
                        print(f"  [FALL] Pin ID={tid} fell at {m:02d}:{s:02d}  "
                              f"(total down: {detector.fallen_count()})")
                        events_log.append({
                            "pin_id": tid,
                            "fall_timestamp_sec": round(ts, 3),
                            "fall_frame": frame_idx,
                        })

        # Keep last known position for fallen pins that are temporarily occluded
        for tid in fallen_pins:
            if tid not in pin_centroids and tid in detector._last_centroids:
                pin_centroids[tid] = detector._last_centroids[tid]

        # --- Render overlays ---
        draw_frame(
            frame,
            frame_idx,
            fps,
            fallen_pins,
            pin_centroids,
            detector.fallen_count(),
            total_frames,
        )

        writer.write(frame)

        # Progress indicator every second
        if frame_idx % max(1, int(fps)) == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  {pct:5.1f}%  frame {frame_idx}/{total_frames}", end="\r")

        frame_idx += 1

    cap.release()
    writer.release()

    # --- Save events log ---
    log_path = output_path.parent / "events_log.json"
    summary = {
        "input_video": str(input_path),
        "total_frames_processed": frame_idx,
        "fps": fps,
        "total_duration_sec": round(frame_idx / fps, 2),
        "total_pins_knocked_down": detector.fallen_count(),
        "pin_fall_events": events_log,
    }
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\n{'='*50}")
    print(f"  Output video : {output_path}")
    print(f"  Events log   : {log_path}")
    print(f"  SCORE        : {detector.fallen_count()} pin(s) knocked down")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
