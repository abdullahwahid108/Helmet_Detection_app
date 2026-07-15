# Annotation Log

## Final Counts by Split

| Split | # Images | # Labels (boxes) |
|---|---|---|
| train | 371 | 2,118 |
| val | 104 | 485 |
| test | 53 | 171 |
| **Total** | **528** | **2,774** |

Number of classes: 2 (`helmet`, `no_helmet`) — a third `person` class was
initially included via auto-labeling but was deliberately dropped since it
was never manually annotated and wasn't needed for the project's core
safety-compliance goal.

## Per-class box counts (train split)
| Class | Count |
|---|---|
| helmet | 1,262 |
| no_helmet | 856 |

## Per-class box counts (val split)
| Class | Count |
|---|---|
| helmet | 353 |
| no_helmet | 132 |

## Per-class box counts (test split)
| Class | Count |
|---|---|
| helmet | 110 |
| no_helmet | 61 |

## Annotation Process & Tools
- Initially attempted with LabelImg (local, manual) — abandoned after a
  recurring PyQt5 compatibility bug (`canvas.py` float/int TypeError) made
  the tool unusable after multiple patch attempts, combined with the
  workload of manually annotating 3 classes across 568 images.
- Switched to Roboflow (web-based): images uploaded unlabeled, annotated
  using a mix of manual boxing and Label Assist (SAM3 zero-shot
  auto-labeling) for faster first-pass drafts, followed by manual review
  and correction of every image.

## QA notes / issues found and fixed
- SAM3 auto-labeling exported some annotations as segmentation polygons
  instead of plain bounding boxes; fixed with a custom script
  (`fix_polygon_labels.py`) that converts polygon coordinates to bounding
  boxes via min/max point extraction (2,666 lines converted).
- A class-index mismatch was discovered where `data.yaml` and the actual
  label files disagreed on which index meant `helmet` vs `no_helmet` —
  resolved by visually inspecting labeled images directly
  (`check_one_image.py`) rather than trusting assumed ordering.
- Initial no_helmet labeling was severely under-represented (1 box out of
  1,263 total) due to SAM3's weakness at detecting "absence of an object."
  Fixed by manually adding ~856 additional no_helmet boxes directly in
  Roboflow before final export.
- A stray single-box `Person` class (capitalized, likely an accidental
  duplicate/misclick) was found via Roboflow's class list page and removed.
- Labeling rule established for consistency: any non-protective headwear
  (caps, hoods, bare heads) counts as `no_helmet`; age (including children)
  does not exempt a person from labeling.