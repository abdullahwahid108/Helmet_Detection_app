import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

CLASS_NAMES = ["helmet", "no_helmet"]
COLORS = [(0, 200, 0), (0, 0, 220), (200, 150, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    args = ap.parse_args()

    img_path = Path(args.image)
    lbl_path = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if not lbl_path.exists():
        print(f"No label file found at {lbl_path}")
        return

    for line in lbl_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split())
        cls = int(cls)
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class{cls}"
        cv2.putText(img, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(img_path.name)
    plt.axis("off")
    out_path = Path("docs") / "single_image_check.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()