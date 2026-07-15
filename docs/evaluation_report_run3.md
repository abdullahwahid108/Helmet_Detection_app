# Evaluation Report — run3 (larger model)

## Settings
- Model: yolov8s (upgraded from yolov8n)
- Epochs: 50, Image size: 640, Batch: 16, LR0: 0.01
- Change from run2: larger base model only; same hard-negative
  backgrounds retained.
- Note: this run took ~46 hours wall-clock due to apparent system
  throttling/sleep during several epochs (isolated epochs took up to
  1380s vs. a normal ~30s) — not representative of yolov8s's typical
  training speed on this hardware.

## Metrics (test set)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| helmet | 0.807 | 0.635 | 0.685 | 0.488 |
| no_helmet | 0.701 | 0.667 | 0.707 | 0.392 |
| **Overall** | **0.754** | **0.656** | **0.696** | **0.440** |

## Confusion Matrix (normalized)
- True helmet → predicted helmet: 65% (down from 72% in run1)
- True no_helmet → predicted no_helmet: 84% (up from 75% in run1)
- True background → predicted helmet: 61% (essentially unchanged from run1's 62%)
- True background → predicted no_helmet: 39% (essentially unchanged from run1's 38%)

## Sample Predictions Analysis
Tested on out-of-distribution images (different source/quality than
training data):
- Missed multiple clearly-visible yellow hard hats in two separate images.
- Falsely detected "helmet" in an empty sky region with 0.73 confidence
  and inside a metal pipe interior with 0.85 confidence.
- Correctly detected several helmets in a busy multi-person construction
  scene, though with some misses on bare heads in the same image.

## Strengths
- Overall mAP50 (0.696) and overall recall (0.656) are the best of the
  three runs.
- no_helmet recall improved substantially (0.75 → 0.84), suggesting the
  larger model has more capacity to learn this harder-to-detect class.

## Weaknesses
- **This is a tradeoff, not a uniform improvement**: helmet recall
  actually decreased (0.72 → 0.65) even as no_helmet improved. With a
  small dataset (~371 training images), increased model capacity
  reallocates learned decision boundaries rather than uniformly improving
  every class.
- **The background false-positive/hallucination problem persisted
  essentially unchanged across all three experiments** (~60-62% for
  helmet, ~38-41% for no_helmet), regardless of negative-example
  augmentation or model size — strong evidence this is a dataset-scale
  limitation, not a fixable architecture/hyperparameter issue.
- Clear generalization gap on out-of-distribution images (different
  compression, lighting, camera quality, possibly underrepresented helmet
  colors) — missed real helmets and hallucinated detections on completely
  unfamiliar backgrounds (sky, pipe interior).

## Conclusion
Selected as the final model for the application based on best overall
mAP50/recall, while explicitly acknowledging the unresolved background
hallucination issue and out-of-distribution generalization gap as honest,
documented limitations of a ~528-image dataset.