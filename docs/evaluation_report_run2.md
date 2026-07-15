# Evaluation Report — run2 (+ hard-negative backgrounds)

## Settings
- Model: yolov8n (same as run1)
- Epochs: 50, Image size: 640, Batch: 16, LR0: 0.01
- Change from run1: added ~20-40 hard-negative background images
  (coiled wire, rocky/mining textures, machinery) with empty label files,
  directly targeting the false-positive shortcut found in run1.

## Metrics (final training-epoch validation)

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **Overall** | 0.726 | 0.582 | 0.660 | 0.419 |

## Confusion Matrix (normalized)
- True background → predicted helmet: 59% (vs. 62% in run1 — marginal change)
- True background → predicted no_helmet: 41% (vs. 38% in run1 — slightly worse)

## Sample Predictions Analysis
Directly re-checked the specific images that fooled run1:
- The coiled-wire image: no longer detected as helmet — **but a
  different nearby object in the same image was newly detected as
  helmet instead.**
- The machinery-tip image: no longer falsely detected.
- A real helmet that run1 correctly detected was missed in run2.

## Strengths
- The exact false positives shown to the model during training (as
  negative examples) were successfully suppressed.

## Weaknesses
- **Classic "whack-a-mole" pattern**: the model appears to have memorized
  that the *specific* negative examples shown aren't helmets, rather than
  learning the general concept of what a helmet actually looks like. A new,
  different false positive appeared elsewhere, and a previously-correct
  detection was lost — a precision/recall tradeoff rather than a net gain.
- Aggregate metrics did not improve over run1, and the core background
  false-positive rate barely moved (59% vs 62%), suggesting ~20-40
  negative examples was too small a nudge relative to ~371 positive
  training images to meaningfully shift the model's learned boundary.

## Next steps to improve
- Try a larger model (more capacity to learn genuine features rather than
  shortcuts) → tested in run3.
- Longer-term: a much larger and more varied set of hard-negative examples
  (hundreds, not dozens) would likely be needed to properly fix this.