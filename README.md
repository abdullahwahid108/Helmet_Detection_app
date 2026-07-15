# Helmet Detection — End-to-End YOLO Project

## Project Summary (Completed)

A YOLOv8-based helmet safety-compliance detector, built end-to-end: dataset
collection from 3 Kaggle sources + stock photos + video-frame extraction
(568 raw → 528 final images), manual annotation via Roboflow (2 classes:
`helmet`, `no_helmet`), three documented training experiments, and a
full-featured Streamlit application (image/video/webcam detection,
confidence threshold slider, detection stats, export, dashboard, detection
history, dark theme).

**Final model**: yolov8s, trained for 50 epochs — overall mAP50 0.696,
recall 0.656 on held-out test data (see `docs/experiment_summary.md` for
full experiment comparison and honest limitations).

**Key documentation:**
- `docs/dataset_log.md` — data sources and coverage
- `docs/annotation_log.md` — annotation process, counts, and issues found/fixed
- `docs/split_documentation.md` — train/val/test split details
- `docs/training_log.md` — auto-generated log of all training runs
- `docs/evaluation_report_run1.md`, `run2.md`, `run3.md` — per-run analysis
- `docs/experiment_summary.md` — final comparison and conclusions

---

Classes: `helmet`, `no_helmet`

This repo is a full scaffold. Follow the steps in order — each step has a
script/template ready for you; you just need to supply the data.

---

## Step 1 — Build the Dataset (300–1000 images)

Put raw images (no labels yet) into `data/raw/images/`.

**If sourcing purely from video frame extraction (`extract_frames.py`):**
- Use **5–8+ distinct source videos** (different locations, times of day, camera
  distances/angles) — not one long video. A single video gives you many frames
  but almost no real diversity; augmentation cannot invent new backgrounds or
  lighting conditions, it can only vary the ones you actually captured.
- Use a consistent `--prefix` per source video (e.g. `site1`, `site2`, `moto1`)
  so `split_dataset.py --group-by-prefix` can split by video, not by frame —
  this prevents near-duplicate frames from leaking across train/val/test and
  inflating your test metrics.
- Ultralytics/YOLO already applies augmentation automatically during training
  (mosaic, HSV jitter, flips, scaling) — you don't need to do this yourself
  for general robustness.
- Use `scripts/augment_dataset.py` (offline, bbox-aware) only for targeted
  cases: balancing an under-represented class (e.g. `no_helmet`) or simulating
  a condition you couldn't film. Run it on the train split only, never on
  val/test.

**Sourcing strategy (mix these so your model generalizes):**
- Construction-site photos/videos (helmets vs bare heads, hi-vis vests nearby for distraction)
- Motorcyclist/traffic footage (street-level angle, motion blur, helmets of different colors/types)
- Public datasets to seed/augment (check licenses before use): Kaggle "Hard Hat Detection", "Safety Helmet Detection" datasets, Roboflow Universe "helmet detection" projects
- Your own phone photos: vary background (indoor/outdoor/urban/rural), lighting (daylight, dusk, indoor fluorescent, backlit), angle (eye-level, top-down CCTV-style, low-angle), occlusion (hand covering face, crowd overlap), scale (close-up vs far/crowd shot)

**Target minimum coverage checklist** (log this in `docs/dataset_log.md`):
- [ ] ≥150 images with helmet present
- [ ] ≥100 images with bare head (no_helmet) present
- [ ] ≥5 distinct backgrounds
- [ ] ≥3 lighting conditions
- [ ] ≥3 camera angles
- [ ] Some occluded examples
- [ ] Some far/small-scale examples

---

## Step 2 — Split the Data

Once raw images are collected, run:

```bash
python scripts/split_dataset.py --source data/raw --dest data --train 0.7 --val 0.2 --test 0.1
```

**If your images came from video frame extraction**, add `--group-by-prefix`
so entire videos (not individual frames) get assigned to a single split —
this avoids near-duplicate frames leaking between train and test:

```bash
python scripts/split_dataset.py --source data/raw --dest data --train 0.7 --val 0.2 --test 0.1 --group-by-prefix
```

This copies images into `data/images/{train,val,test}` and auto-writes
`docs/split_documentation.md` with exact counts — don't hand-edit that file,
just re-run the script if you add more data.

---

## Step 3 — Annotate

Use **LabelImg**, **CVAT**, or **Roboflow Annotate** (Roboflow is easiest for
beginners and exports directly to YOLO format).

Install LabelImg locally:
```bash
pip install labelImg
labelImg data/images/train data/classes.txt
```

Save labels as YOLO `.txt` format into the matching `data/labels/{split}/` folder
(same filename as the image, `.txt` extension).

`data/classes.txt`:
```
helmet
no_helmet
person
```

**Annotation rules:**
- Draw box tight to the object edge — no extra margin
- Label every visible instance, even small/partial ones (unless <10% visible)
- Be consistent: if the head is covered by any hard-hat-like object, it's `helmet`

After annotating, run:
```bash
python scripts/visualize_annotations.py --split train --n 12
```
This renders a grid of random annotated images so you can visually QA your boxes
before training.

Then fill in `docs/annotation_log.md` with:
1. Total number of images
2. Total number of labels (bounding boxes)
3. Number of classes (3)

---

## Step 4 — Train YOLO

Edit `data.yaml` (already scaffolded) to point at your split folders, then:

```bash
pip install ultralytics
python scripts/train.py --epochs 100 --imgsz 640 --batch 16 --lr0 0.01 --name run1
```

Run this multiple times varying `--epochs`, `--imgsz`, `--batch`, `--lr0`.
Every run auto-appends its settings + final metrics to `docs/training_log.md` —
this is your experiment record, keep it.

Suggested experiments to actually learn something (not just get a working model):
| Run | epochs | imgsz | batch | lr0 | Purpose |
|---|---|---|---|---|---|
| run1 | 50 | 640 | 16 | 0.01 | baseline |
| run2 | 100 | 640 | 16 | 0.01 | does more training help or overfit? |
| run3 | 100 | 416 | 32 | 0.01 | smaller imgsz, bigger batch — speed vs accuracy |
| run4 | 100 | 640 | 16 | 0.001 | lower LR — smoother convergence? |

---

## Step 5 — Evaluate

```bash
python scripts/evaluate.py --weights runs/detect/run1/weights/best.pt --name run1
```

This generates, into `docs/evaluation_run1/`:
- `metrics.json` (precision, recall, mAP50, mAP50-95 per class + overall)
- `confusion_matrix.png`
- `sample_predictions/` (annotated sample outputs)

Then write your analysis into `docs/evaluation_report_template.md` — specifically:
- Which class has weakest recall (likely `no_helmet` — why? fewer examples? more visual variety?)
- Where does the model confuse `helmet` vs `person`?
- Does performance drop on small/occluded objects in your sample predictions?

Pick your best run's weights and copy them to `app/model/best.pt` for the app.

---

## The App

See `app/app.py` (Streamlit). Run with:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Features: image upload, video upload, live webcam, confidence threshold slider,
detection stats (count/time/confidence), export annotated image, plus bonus
dashboard + detection history + dark theme.

Put your trained weights at `app/model/best.pt` before running.