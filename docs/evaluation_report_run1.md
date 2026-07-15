# Evaluation Report — run1 (baseline)

## Settings
- Model: yolov8n
- Epochs: 50, Image size: 640, Batch: 16, LR0: 0.01 (optimizer auto-selected AdamW)

## Metrics (test set)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| helmet | 0.870 | 0.607 | 0.748 | 0.524 |
| no_helmet | 0.936 | 0.720 | 0.813 | 0.516 |
| **Overall** | **0.903** | **0.664** | **0.780** | **0.520** |

## Confusion Matrix (normalized)
- True helmet → predicted helmet: 72%
- True no_helmet → predicted no_helmet: 75%
- True background → predicted helmet: 62% (false positive)
- True background → predicted no_helmet: 38% (false positive)
- True no_helmet → predicted helmet: 10% (cross-class confusion)

## Sample Predictions Analysis
- Strong precision overall — when the model detects something, it's usually right.
- Clear weakness: false positives on background clutter. Specific
  failure cases found: coiled wire/rope detected as helmet (0.73
  confidence), rocky machinery tip detected as helmet (0.93 confidence).
- Recall (~0.66) indicates roughly a third of real objects are missed,
  likely concentrated in small/distant/occluded cases given the dataset's
  deliberate inclusion of these hard scenarios.

## Strengths
- High precision on both classes (>0.87) — few false alarms on clearly
  visible helmets/heads.
- Balanced performance between helmet and no_helmet classes, confirming
  the class-imbalance fix (adding ~856 no_helmet boxes) was effective.

## Weaknesses
- Model appears to use a shape/position shortcut ("round blob near head
  height = helmet") rather than learning true helmet-specific visual
  features (material, curvature, strap) — evidenced by confident false
  positives on visually similar but unrelated objects.
- Recall is the weaker of the two headline metrics, missing roughly a
  third of real objects.

## Next steps to improve
- Add hard-negative background examples (visually similar clutter with no
  real helmet/head present) to directly target the false-positive
  shortcut-learning problem → tested in run2.