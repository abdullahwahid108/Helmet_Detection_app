"""
augment_dataset.py

Offline, bounding-box-aware augmentation. Use this SPARINGLY and with intent —
YOLO/ultralytics already augments automatically during training (mosaic, HSV
jitter, flips, scaling). This script is for two specific situations:

  1. Balancing classes: generate extra augmented copies of images that
     contain an under-represented class (e.g. `no_helmet`), so the class
     distribution going into training is less skewed.
  2. Simulating conditions you couldn't film: brightness/contrast shifts,
     motion blur, etc.

Run this on your TRAIN split only (never augment val/test — you evaluate on
real, unaltered data).

Setup:
    pip install albumentations opencv-python

Usage (general augmentation pass over all training images):
    python scripts/augment_dataset.py --images data/images/train \
        --labels data/labels/train --out-images data/images/train \
        --out-labels data/labels/train --copies 1

Usage (only augment images containing a specific class, e.g. to balance
no_helmet which is class index 1 in data.yaml):
    python scripts/augment_dataset.py --images data/images/train \
        --labels data/labels/train --out-images data/images/train \
        --out-labels data/labels/train --copies 3 --only-class 1
"""
import argparse
import random
from pathlib import Path
import cv2
import albumentations as A


def load_yolo_labels(lbl_path):
    boxes, class_ids = [], []
    if not lbl_path.exists():
        return boxes, class_ids
    for line in lbl_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = map(float, line.split())
        boxes.append([cx, cy, w, h])
        class_ids.append(int(cls))
    return boxes, class_ids


def save_yolo_labels(lbl_path, boxes, class_ids):
    lines = []
    for box, cls in zip(boxes, class_ids):
        cx, cy, w, h = box
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build_transform():
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.RandomShadow(p=0.15),
            A.Affine(scale=(0.85, 1.15), translate_percent=(0.0, 0.05),
                     rotate=(-8, 8), p=0.5),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_ids"], min_visibility=0.2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--out-images", type=str, required=True)
    ap.add_argument("--out-labels", type=str, required=True)
    ap.add_argument("--copies", type=int, default=1,
                     help="augmented copies to generate per qualifying image")
    ap.add_argument("--only-class", type=int, default=None,
                     help="if set, only augment images that contain at least "
                          "one box of this class index (see data.yaml for indices)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    out_img_dir = Path(args.out_images)
    out_lbl_dir = Path(args.out_labels)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform()
    img_paths = sorted([p for p in img_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    generated = 0
    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes, class_ids = load_yolo_labels(lbl_path)
        if not boxes:
            continue
        if args.only_class is not None and args.only_class not in class_ids:
            continue

        image = cv2.imread(str(img_path))

        for i in range(args.copies):
            augmented = transform(image=image, bboxes=boxes, class_ids=class_ids)
            aug_img = augmented["image"]
            aug_boxes = augmented["bboxes"]
            aug_classes = augmented["class_ids"]
            if not aug_boxes:  # all boxes got clipped out, skip this copy
                continue

            out_name = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(out_img_dir / f"{out_name}{img_path.suffix}"), aug_img)
            save_yolo_labels(out_lbl_dir / f"{out_name}.txt", aug_boxes, aug_classes)
            generated += 1

    print(f"Generated {generated} augmented image/label pairs in {out_img_dir}")
    print("Re-run scripts/visualize_annotations.py to sanity-check the "
          "augmented boxes before training.")


if __name__ == "__main__":
    main()
