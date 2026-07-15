"""
visualize_annotations.py

Renders a grid of randomly-sampled annotated images from a given split so you
can visually QA your bounding boxes before training. Also prints/writes
dataset stats: image count, label (box) count, class count and per-class
counts -- feed these numbers into docs/annotation_log.md.

Usage:
    python scripts/visualize_annotations.py --split train --n 12
"""
import argparse
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

CLASS_NAMES = ["helmet", "no_helmet"]
COLORS = [(0, 200, 0), (0, 0, 220), (200, 150, 0)]


def load_labels(lbl_path, w, h):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split())
        cls = int(cls)
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--data-root", type=str, default="data")
    args = ap.parse_args()

    img_dir = Path(args.data_root) / "images" / args.split
    lbl_dir = Path(args.data_root) / "labels" / args.split

    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in
                         {".jpg", ".jpeg", ".png"}])
    if not img_paths:
        print(f"No images found in {img_dir}")
        return

    total_boxes = 0
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    for p in img_paths:
        lbl_path = lbl_dir / (p.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                if line.strip():
                    cls = int(line.split()[0])
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                    total_boxes += 1

    print(f"[{args.split}] images: {len(img_paths)} | total boxes: {total_boxes}")
    for cls, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {class_counts.get(cls, 0)}")

    sample = random.sample(img_paths, min(args.n, len(img_paths)))
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, img_path in zip(axes, sample):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        for cls, x1, y1, x2, y2 in load_labels(lbl_path, w, h):
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, CLASS_NAMES[cls], (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        ax.imshow(img)
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = Path("docs") / f"annotation_qa_{args.split}.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"Saved QA grid to {out_path}")


if __name__ == "__main__":
    main()
