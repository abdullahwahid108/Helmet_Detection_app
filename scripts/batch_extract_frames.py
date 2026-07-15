"""
batch_extract_frames.py

Runs frame extraction (with near-duplicate filtering) over EVERY video in a
folder at once. Each video's filename (without extension) becomes its
prefix automatically, so you don't need to run the command manually for
every single clip.

Usage:
    python scripts/batch_extract_frames.py --videos-dir "C:\\path\\to\\stock video" \
        --out data/raw/images --every-n-seconds 1.0
"""
import argparse
import re
from pathlib import Path
import cv2
from PIL import Image
import imagehash

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def sanitize_prefix(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return clean or "vid"


def extract_one_video(video_path, out_dir, every_n_seconds, hash_threshold, prefix):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps * every_n_seconds), 1)

    frame_idx = 0
    saved_idx = 0
    skipped_dupes = 0
    last_hash = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            h = imagehash.phash(pil_img)

            if last_hash is not None and (h - last_hash) < hash_threshold:
                skipped_dupes += 1
            else:
                fname = f"{prefix}_{saved_idx:04d}.jpg"
                cv2.imwrite(str(out_dir / fname), frame)
                saved_idx += 1
                last_hash = h

        frame_idx += 1

    cap.release()
    return saved_idx, skipped_dupes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/raw/images")
    ap.add_argument("--every-n-seconds", type=float, default=1.0)
    ap.add_argument("--hash-threshold", type=int, default=5)
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted([p for p in videos_dir.iterdir()
                           if p.suffix.lower() in VIDEO_EXTS])

    if not video_paths:
        print(f"No video files found in {videos_dir}")
        return

    print(f"Found {len(video_paths)} videos. Extracting...")
    total_saved = 0
    for video_path in video_paths:
        prefix = sanitize_prefix(video_path.stem)
        saved, skipped = extract_one_video(
            video_path, out_dir, args.every_n_seconds, args.hash_threshold, prefix
        )
        total_saved += saved
        print(f"  {video_path.name} -> prefix '{prefix}': saved {saved} frames "
              f"(skipped {skipped} near-duplicates)")

    print(f"\nDone. Total frames saved across all videos: {total_saved}")
    print(f"All frames written to {out_dir}")


if __name__ == "__main__":
    main()