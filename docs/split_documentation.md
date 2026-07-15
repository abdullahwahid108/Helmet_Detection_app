# Dataset Split Documentation

Final split performed via Roboflow's dataset versioning (not the local
split_dataset.py script), with manual reassignment of video-frame-derived
images to ensure frames from the same source video stayed within a single
split (preventing near-duplicate leakage between train/val/test).

| Split | Ratio (approx) | Image Count |
|---|---|---|
| train | 70.3% | 371 |
| val | 19.7% | 104 |
| test | 10.0% | 53 |
| **Total** | | **528** |

## Notes
- Video-frame images (prefixed `vid1_`, `vid2_`, etc., and similar) were
  manually checked and grouped so that all frames from a single source
  video landed in the same split, avoiding train/test leakage from
  near-identical consecutive frames.
- Kaggle-sourced and stock-photo images (independent single photos, no
  video correlation) were split randomly by Roboflow's standard split
  assignment.
- Class balance across splits is reasonably consistent: roughly 60-70%
  helmet / 30-40% no_helmet in each of train, val, and test.