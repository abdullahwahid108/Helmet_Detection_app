"""
split_dataset.py

Splits raw (image, label) pairs into train/val/test sets and copies them
into the YOLO folder structure. Also writes docs/split_documentation.md
with exact counts so the split is documented, not just implied.

Expected input layout:
    data/raw/images/*.jpg (or .png)
    data/raw/labels/*.txt   (YOLO format, same basename as image)

Usage:
    python scripts/split_dataset.py --source data/raw --dest data \
        --train 0.7 --val 0.2 --test 0.1 --seed 42
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from datetime import datetime

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_pairs(source: Path):
    img_dir = source / "images"
    lbl_dir = source / "labels"
    pairs = []
    missing_labels = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            missing_labels.append(img_path.name)
    return pairs, missing_labels


def group_key(img_path: Path):
    """Group frames from the same source video together (filename prefix
    before the last underscore, e.g. 'site1_0042.jpg' -> 'site1') so that
    near-duplicate frames from one video never get split across train/val/test."""
    stem = img_path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="data/raw")
    ap.add_argument("--dest", type=str, default="data")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-by-prefix", action="store_true",
                     help="Split by filename prefix (source video) instead of by "
                          "individual frame. USE THIS if your images come from video "
                          "frame extraction, to avoid near-duplicate leakage across "
                          "train/val/test. Requires consistent prefixes from "
                          "extract_frames.py (e.g. site1_0001.jpg, site1_0002.jpg...).")
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    source = Path(args.source)
    dest = Path(args.dest)
    pairs, missing_labels = find_pairs(source)

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images have no matching label file "
              f"and will be SKIPPED. Annotate them first. First few: {missing_labels[:5]}")

    random.seed(args.seed)

    if args.group_by_prefix:
        groups = defaultdict(list)
        for img_path, lbl_path in pairs:
            groups[group_key(img_path)].append((img_path, lbl_path))
        group_names = list(groups.keys())
        random.shuffle(group_names)
        print(f"Found {len(group_names)} source groups (videos): {group_names}")

        n_groups = len(group_names)
        n_train_g = max(int(n_groups * args.train), 1)
        n_val_g = max(int(n_groups * args.val), 1) if n_groups > 2 else 0
        train_groups = group_names[:n_train_g]
        val_groups = group_names[n_train_g:n_train_g + n_val_g]
        test_groups = group_names[n_train_g + n_val_g:]

        splits = {
            "train": [pair for g in train_groups for pair in groups[g]],
            "val": [pair for g in val_groups for pair in groups[g]],
            "test": [pair for g in test_groups for pair in groups[g]],
        }
        print(f"train groups: {train_groups}\nval groups: {val_groups}\ntest groups: {test_groups}")
    else:
        random.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        splits = {
            "train": pairs[:n_train],
            "val": pairs[n_train:n_train + n_val],
            "test": pairs[n_train + n_val:],
        }

    n = len(pairs)

    for split_name, split_pairs in splits.items():
        img_out = dest / "images" / split_name
        lbl_out = dest / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    # Write documentation
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    doc_path = docs_dir / "split_documentation.md"
    with open(doc_path, "w") as f:
        f.write("# Dataset Split Documentation\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"Random seed: {args.seed}\n\n")
        f.write(f"Split mode: {'grouped by source video/prefix (leakage-safe)' if args.group_by_prefix else 'random per-image'}\n\n")
        f.write(f"Total valid image/label pairs found: {n}\n")
        f.write(f"Images skipped (missing labels): {len(missing_labels)}\n\n")
        f.write("| Split | Ratio | Image Count |\n|---|---|---|\n")
        f.write(f"| train | {args.train} | {len(splits['train'])} |\n")
        f.write(f"| val | {args.val} | {len(splits['val'])} |\n")
        f.write(f"| test | {args.test} | {len(splits['test'])} |\n")

    print(f"Done. {n} pairs split -> train:{len(splits['train'])} "
          f"val:{len(splits['val'])} test:{len(splits['test'])}")
    print(f"Documentation written to {doc_path}")


if __name__ == "__main__":
    main()
