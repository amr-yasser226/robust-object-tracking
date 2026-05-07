import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on target videos.")
    parser.add_argument("--model", type=str, default="../models/trained/best.pt", help="Path to the trained YOLOv8 weights.")
    parser.add_argument("--input_dir", type=str, default="./inputs", help="Directory containing input videos.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the inference results.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for bounding box predictions.")
    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model weights not found at {model_path}")
        print("Please ensure the model path is correct or run this script from the 'inference/' directory.")
        return

    # Load the model
    print(f"[INFO] Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input directory {input_dir} does not exist.")
        return

    # Find all video files
    supported_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in supported_extensions]

    if not video_files:
        print(f"[WARNING] No video files found in {input_dir}. Please place your videos there.")
        return

    print(f"[INFO] Found {len(video_files)} video(s) for processing.")

    # Run inference
    for video_file in video_files:
        print(f"\n[INFO] Processing video: {video_file.name}...")
        
        # model.predict automatically handles video streams, frame-by-frame inference,
        # drawing bounding boxes, and compiling the output video if save=True.
        # We explicitly set the project and name so ultralytics doesn't nest it deeply in runs/predict
        results = model.predict(
            source=str(video_file),
            conf=args.conf,
            save=True,
            project=args.output_dir,
            name=video_file.stem,  # Creates a folder named after the video
            exist_ok=True,
            verbose=False
        )
        
        print(f"[INFO] Successfully processed {video_file.name}.")
        print(f"[INFO] Output saved in {Path(args.output_dir) / video_file.stem}")

    print("\n[SUCCESS] All inference tasks completed.")

if __name__ == "__main__":
    main()
