# Dataset Collection Log

## Sources
| Source | # Images | Notes |
|---|---|---|
| Kaggle: hard-hat-detection (andrewmvd) | 150 | Random subset of ~5000 total; images only, original annotations discarded |
| Kaggle: helmet-detection (andrewmvd) | 100 | 764 total images, random subset used |
| Kaggle: rider-with-helmet-without-helmet (aneesarom) | 100 | ~120 total images, most of the set used |
| Stock/traffic photos (Unsplash, Pexels, traffic images) | 91 | Free-use stock sources; supplements motorcyclist/construction variety |
| Video frame extraction (10 short clips, mixed Pexels stock video + own selection) | 203 | Extracted via extract_frames.py / batch_extract_frames.py with near-duplicate frame filtering (perceptual hashing) |
| **Total raw collected** | **568** | |
| Removed during visual QA (blurry/unclear/broken) | ~76 | Manually reviewed and dropped before/during annotation |
| **Final dataset size** | **528** | After Roboflow annotation, class-index fixes, and cleanup |

## Coverage Checklist
- [x] ≥150 images with helmet present — confirmed via final label counts (1262 helmet boxes total across splits)
- [x] ≥100 images with bare head (no_helmet) present — confirmed via final label counts (856 no_helmet boxes total across splits, after a mid-project fix to address initial under-labeling)
- [x] ≥5 distinct backgrounds — construction sites, street/traffic scenes, indoor stock settings, night/rain traffic footage, mining/tunnel scenes (from video sources), misc stock backgrounds
- [x] ≥3 lighting conditions — daylight, indoor/stock lighting, night + rain (low light, glare, wet-surface reflections)
- [x] ≥3 camera angles — street-level traffic camera angle, close-up stock/Kaggle portraits, varied handheld/stock-video angles
- [x] Occlusion examples included — partially obscured motorcyclists in traffic footage, crowd overlap in construction/stock images
- [x] Scale variation included — close-up portraits vs. far/wide traffic and construction scene shots

## Total raw images collected: 568
## Final images used after cleanup: 528