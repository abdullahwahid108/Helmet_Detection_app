import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True)
    ap.add_argument("--prefix", type=str, default="negbg")
    ap.add_argument("--dest-images", type=str, default="data/images/train")
    ap.add_argument("--dest-labels", type=str, default="data/labels/train")
    args = ap.parse_args()

    source = Path(args.source)
    dest_img = Path(args.dest_images)
    dest_lbl = Path(args.dest_labels)
    dest_img.mkdir(parents=True, exist_ok=True)
    dest_lbl.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS])

    if not img_paths:
        print(f"No images found in {source}")
        return

    for i, img_path in enumerate(img_paths):
        new_name = f"{args.prefix}_{i:04d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dest_img / new_name)
        (dest_lbl / f"{args.prefix}_{i:04d}.txt").write_text("")

    print(f"Added {len(img_paths)} background-only images to {dest_img}")
    print(f"Created matching empty label files in {dest_lbl}")


if __name__ == "__main__":
    main()