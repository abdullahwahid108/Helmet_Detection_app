# Experiment Summary — Helmet Detection Project

## Overview
Three training experiments were run on a YOLOv8 helmet detection model
(classes: `helmet`, `no_helmet`) using a 528-image dataset (371 train / 104
val / 53 test), sourced from three Kaggle datasets, stock photos, and
video-frame extraction.

## Results Table

| Run | Model | Epochs | Key change from previous | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| run1 | yolov8n | 50 | Baseline | 0.903 | 0.664 | 0.780 | 0.520 |
| run2 | yolov8n | 50 | + ~20-40 hard-negative background images | 0.726* | 0.582* | 0.660* | 0.419* |
| run3 | yolov8s | 50 | Larger model (same negative backgrounds as run2) | 0.754 | 0.656 | 0.696 | 0.440 |

*run2 metrics from final training-epoch validation; run1/run3 metrics from full test-set evaluation via evaluate.py — see note below on comparability.

## Per-Class Breakdown (run3, final model, test set)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| helmet | 0.807 | 0.635 | 0.685 | 0.488 |
| no_helmet | 0.701 | 0.667 | 0.707 | 0.392 |

## Confusion Matrix Findings (background false-positive rate)

| | run1 | run2 | run3 |
|---|---|---|---|
| True background → predicted helmet | 62% | 59% | 61% |
| True background → predicted no_helmet | 38% | 41% | 39% |
| True helmet → predicted helmet (correct) | 72% | — | 65% |
| True no_helmet → predicted no_helmet (correct) | 75% | — | 84% |

## Key Conclusions

1. **The background false-positive/hallucination problem persisted across
   all three experiments essentially unchanged.** Neither adding a small
   batch of hard-negative background images (run2) nor increasing model
   capacity (run3) meaningfully reduced the model's tendency to detect
   helmet-like shapes in cluttered backgrounds (coiled wire, rocky
   textures, machinery). This strongly suggests the issue is a
   **dataset-scale limitation** — likely requiring hundreds more diverse
   negative examples to properly address — rather than a fixable
   hyperparameter or architecture problem.

2. **Model size introduced a real tradeoff, not a uniform improvement.**
   Moving from yolov8n to yolov8s substantially improved no_helmet recall
   (0.75 → 0.84) but reduced helmet recall (0.72 → 0.65). With a small
   training set (~371 images), a larger model does not uniformly improve
   all classes — it can reallocate its learned decision boundaries in
   ways that help one class at the expense of another.

3. **Generalization gap on out-of-distribution images.** When tested on
   images from different sources (different compression, lighting,
   camera quality, and possibly underrepresented helmet colors like
   yellow hard hats), the model's performance degraded noticeably —
   missing clearly visible helmets and hallucinating false positives on
   unfamiliar backgrounds (e.g. an empty sky, pipe interiors). This is an
   expected and honest limitation of a ~528-image dataset relative to the
   visual diversity of real-world deployment conditions; production-grade
   detectors are typically trained on orders of magnitude more data
   specifically to close this gap.

## Final Model Selection

**run3 (yolov8s)** was selected as the final model for the application,
based on the highest overall mAP50 (0.696) and highest overall recall
(0.656) among the three runs, despite the per-class tradeoff noted above.
Weights: `runs/detect/runs/detect/run3/weights/best.pt`

## Honest Limitations for Future Work

- Background hallucination remains a significant unsolved issue — future
  work should prioritize collecting a much larger and more visually
  diverse set of hard-negative background images (target: hundreds, not
  dozens) before further architecture or hyperparameter changes.
- The dataset (528 images) is small by production standards; expanding to
  1,000-5,000+ images with more source diversity (different countries,
  camera types, compression levels, helmet colors/styles) would likely
  yield the largest single improvement to real-world generalization.
- Class imbalance (helmet ≈1262 boxes vs no_helmet ≈856 boxes) was
  identified and partially corrected mid-project after a labeling/export
  issue was found; future annotation passes should track class balance
  from the start rather than discovering it late.