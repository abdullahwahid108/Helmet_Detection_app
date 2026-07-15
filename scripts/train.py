"""
train.py

Trains a YOLOv8 model on the helmet dataset and appends the run's settings +
final metrics to docs/training_log.md, so every experiment is recorded
automatically -- you don't have to remember to write it down.

Usage:
    python scripts/train.py --epochs 100 --imgsz 640 --batch 16 --lr0 0.01 --name run1
"""
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data.yaml")
    ap.add_argument("--model", type=str, default="yolov8n.pt",
                     help="base model: yolov8n/s/m/l/x.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--name", type=str, default="run1")
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        name=args.name,
        project="runs/detect",
    )

    # Pull final validation metrics for the log
    metrics = results.results_dict if hasattr(results, "results_dict") else {}

    log_path = Path("docs") / "training_log.md"
    log_path.parent.mkdir(exist_ok=True)
    new_file = not log_path.exists()
    with open(log_path, "a") as f:
        if new_file:
            f.write("# Training Log\n\n")
            f.write("| Run | Timestamp | Base Model | Epochs | ImgSz | Batch | LR0 | "
                    "mAP50 | mAP50-95 | Precision | Recall |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        f.write(
            f"| {args.name} | {datetime.now().isoformat(timespec='seconds')} | "
            f"{args.model} | {args.epochs} | {args.imgsz} | {args.batch} | {args.lr0} | "
            f"{metrics.get('metrics/mAP50(B)', 'NA')} | "
            f"{metrics.get('metrics/mAP50-95(B)', 'NA')} | "
            f"{metrics.get('metrics/precision(B)', 'NA')} | "
            f"{metrics.get('metrics/recall(B)', 'NA')} |\n"
        )

    print(f"Training complete. Weights at runs/detect/{args.name}/weights/best.pt")
    print(f"Logged settings + metrics to {log_path}")


if __name__ == "__main__":
    main()
