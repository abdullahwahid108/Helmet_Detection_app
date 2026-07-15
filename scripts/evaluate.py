"""
evaluate.py

Runs full evaluation of a trained YOLO model on the test set:
- precision, recall, mAP50, mAP50-95 (overall + per-class)
- confusion matrix image
- sample annotated predictions

All outputs go to docs/evaluation_<name>/ so you can write your analysis
straight from these artifacts.

Usage:
    python scripts/evaluate.py --weights runs/detect/run1/weights/best.pt --name run1
"""
import argparse
import json
import shutil
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data", type=str, default="data.yaml")
    ap.add_argument("--name", type=str, default="eval_run")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--n-samples", type=int, default=12,
                     help="number of sample prediction images to save")
    args = ap.parse_args()

    out_dir = Path("docs") / f"evaluation_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    # --- Metrics ---
    metrics = model.val(data=args.data, split=args.split, save_json=True, plots=True)

    results = {
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "per_class": {},
    }
    names = metrics.names
    for i, name in names.items():
        try:
            results["per_class"][name] = {
                "precision": float(metrics.box.p[i]),
                "recall": float(metrics.box.r[i]),
                "mAP50": float(metrics.box.ap50[i]),
                "mAP50-95": float(metrics.box.ap[i]),
            }
        except (IndexError, TypeError):
            pass

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Confusion matrix: ultralytics saves this into its own run dir; copy it over ---
    val_save_dir = Path(metrics.save_dir)
    for candidate in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = val_save_dir / candidate
        if src.exists():
            shutil.copy2(src, out_dir / candidate)

    # --- Sample predictions on test images ---
    data_root = Path("data")
    img_dir = data_root / "images" / args.split
    sample_out = out_dir / "sample_predictions"
    sample_out.mkdir(exist_ok=True)
    img_paths = sorted([p for p in img_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:args.n_samples]
    for img_path in img_paths:
        pred = model.predict(str(img_path), conf=0.25, save=False)[0]
        annotated = pred.plot()
        import cv2
        cv2.imwrite(str(sample_out / img_path.name), annotated)

    print(f"Overall: {results['overall']}")
    print(f"Per-class metrics: {results['per_class']}")
    print(f"All evaluation artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
