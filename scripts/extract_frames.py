"""
extract_frames.py

Extracts frames from a local video file (your own footage, or a video you've
downloaded separately) at a fixed interval, then filters out near-duplicate
frames so you don't waste annotation time on frames that barely differ.

This runs LOCALLY on your machine in VS Code, not in a sandboxed environment,
since it needs access to your video files / webcam-recorded clips.

Setup (run once):
    pip install opencv-python imagehash pillow

If you want to pull a YouTube video first (optional — check the video's
license/usage terms before using it in your dataset):
    pip install yt-dlp
    yt-dlp -f "best[height<=720]" -o "raw_video.mp4" "<YOUTUBE_URL>"

Then extract frames from it:
    python scripts/extract_frames.py --video raw_video.mp4 \
        --out data/raw/images --every-n-seconds 1.5 --prefix site1

Repeat with different --prefix values for each source video so filenames
don't collide.
"""
import argparse
import cv2
from pathlib import Path
from PIL import Image
import imagehash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/raw/images")
    ap.add_argument("--every-n-seconds", type=float, default=1.0,
                     help="sample one frame every N seconds")
    ap.add_argument("--prefix", type=str, default="vid",
                     help="filename prefix, use a different one per source video")
    ap.add_argument("--hash-threshold", type=int, default=5,
                     help="perceptual-hash distance below which a frame is "
                          "considered a near-duplicate of the last kept frame "
                          "(0 = identical, higher = more different required)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps * args.every_n_seconds), 1)

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

            if last_hash is not None and (h - last_hash) < args.hash_threshold:
                skipped_dupes += 1
            else:
                fname = f"{args.prefix}_{saved_idx:04d}.jpg"
                cv2.imwrite(str(out_dir / fname), frame)
                saved_idx += 1
                last_hash = h

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved_idx} frames to {out_dir} "
          f"(skipped {skipped_dupes} near-duplicates).")
    print("Reminder: copy or move these into data/raw/images/ if --out "
          "pointed elsewhere, then annotate them before running split_dataset.py.")


if __name__ == "__main__":
    main()
