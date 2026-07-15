import argparse
import shutil
from pathlib import Path
from datetime import datetime


def polygon_to_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return cx, cy, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", type=str, default="data/labels")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    backup_root = labels_root.parent / f"labels_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(labels_root, backup_root)
    print(f"Backup created at {backup_root}")

    total_files = 0
    total_lines_fixed = 0
    total_lines_kept = 0
    total_lines_skipped = 0

    for txt_path in labels_root.rglob("*.txt"):
        lines = txt_path.read_text().strip().splitlines()
        new_lines = []
        file_changed = False

        for line in lines:
            if not line.strip():
                continue
            tokens = line.split()
            n = len(tokens)

            if n == 5:
                new_lines.append(line.strip())
                total_lines_kept += 1
            elif n > 5 and n % 2 == 1:
                cls = tokens[0]
                coords = list(map(float, tokens[1:]))
                cx, cy, w, h = polygon_to_bbox(coords)
                new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                total_lines_fixed += 1
                file_changed = True
            else:
                print(f"SKIPPING unrecognized line in {txt_path.name}: {n} tokens")
                total_lines_skipped += 1

        if file_changed:
            txt_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            total_files += 1

    print(f"\nDone. Files modified: {total_files}")
    print(f"Polygon lines converted: {total_lines_fixed}")
    print(f"Box lines kept as-is: {total_lines_kept}")
    print(f"Unrecognized skipped: {total_lines_skipped}")


if __name__ == "__main__":
    main()