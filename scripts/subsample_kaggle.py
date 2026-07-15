import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True,
                     help="folder containing the full set of Kaggle images")
    ap.add_argument("--dest", type=str, required=True,
                     help="your data/raw/images folder")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--prefix", type=str, default="hardhat")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-original-annotations", type=str, default=None,
                     help="optional: folder with the original XML annotations")
    ap.add_argument("--answer-key-dest", type=str, default=None,
                     help="required if --keep-original-annotations is set")
    args = ap.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    all_images = sorted([p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if len(all_images) < args.n:
        print(f"WARNING: only {len(all_images)} images found, using all of them.")

    random.seed(args.seed)
    subset = random.sample(all_images, min(args.n, len(all_images)))

    ann_dir = Path(args.keep_original_annotations) if args.keep_original_annotations else None
    answer_key_dest = Path(args.answer_key_dest) if args.answer_key_dest else None
    if ann_dir and answer_key_dest:
        answer_key_dest.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(subset):
        new_name = f"{args.prefix}_{i:04d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dest / new_name)

        if ann_dir and answer_key_dest:
            ann_path = ann_dir / (img_path.stem + ".xml")
            if ann_path.exists():
                shutil.copy2(ann_path, answer_key_dest / f"{args.prefix}_{i:04d}.xml")

    print(f"Copied {len(subset)} images to {dest} with prefix '{args.prefix}_'")
    if ann_dir and answer_key_dest:
        print(f"Original annotations (answer key) copied to {answer_key_dest}")


if __name__ == "__main__":
    main()