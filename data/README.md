# Data

This directory is **git-ignored** except for this file. Images are never committed, for licence
and size reasons.

| Path | Contents | Created by |
|---|---|---|
| `data/raw/kaggle/` | Kaggle download: labels CSV and images | `python -m dr.download` (Phase 1) |
| `data/processed/manifest.csv` | One row per image: ids, label, image statistics, quality flag | `python -m dr.data` (Phase 2) |
| `data/processed/data_quality.json` | Validation and quality report | `python -m dr.data` (Phase 2) |
| `data/processed/image_stats_cache.parquet` | Cache of per-image statistics, so re-runs skip unchanged files | `python -m dr.data` (Phase 2) |
| `data/processed/images_<size>/` | Preprocessed images, cached as lossless PNG | Phase 3 |

Tests never use this directory. They use **synthetic** images from
`scripts/make_synthetic_fixtures.py`.

---

## Dataset card: EyePACS (Kaggle "Diabetic Retinopathy Detection", 2015)

| | |
|---|---|
| **Origin** | EyePACS, a US teleretinal screening programme. Released for the 2015 California Healthcare Foundation / Kaggle competition. |
| **Copy used** | [`tanlikesmath/diabetic-retinopathy-resized`](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) (about 7.8 GB). This is the competition **training** set, resized to a width of 1024 px. |
| **Files used** | `trainLabels.csv` and `resized_train/resized_train/*.jpeg`. The `*_cropped` variant is ignored (`data.exclude_dirs`). |
| **Size** | 35,126 images from 17,563 patients, one left and one right eye each |
| **Labels** | ICDR grade 0–4 from a single clinician per image. The competition notes label noise. |
| **Task label** | `label = 1` (referable) if grade ≥ 2, otherwise 0 |
| **Licence** | Kaggle competition / dataset terms. For research use; do not redistribute images. |

### Measured on the downloaded copy (Phase 2, `data_quality.json`)

| Check | Result |
|---|---|
| Label rows / images found | 35,126 / 35,126 (0 missing, 0 without a label) |
| Duplicate label rows, invalid grades, unparseable names | 0 / 0 / 0 |
| Grade counts 0–4 | 25,810 / 2,443 / 5,292 / 873 / 708, matching the competition |
| Referable share | 19.57% |
| Byte-identical files (same MD5) | 0 |
| Resolution | All 1024 px wide. Two dominant aspect ratios: **3:2** (27,184 images, 21.0% referable) and **4:3** (7,912 images, 14.6% referable), plus 30 near-square images. |
| Left / right referable rate | 19.8% / 19.3% |
| Quality flags | 34,449 ok, 550 `blurred`, 112 `too_dark`, 11 `overexposed`, 4 `no_fundus` (1.9% poor) |

### Known caveats
- **Single source.** All images come from one screening programme in the US. Performance on other
  populations or cameras is unknown until external validation (APTOS 2019, Messidor-2).
- **Camera confounding.** Aspect ratio, a proxy for camera type, is associated with prevalence
  (21.0% vs 14.6% referable). A model could learn the camera instead of the disease, so metrics
  are reported per aspect-ratio group (see the model card).
- **Label noise.** There is one grader per image, and both eyes of a patient often share a grade.
- **Quality flag is a heuristic.** It catches clear capture failures: blur, under- or
  over-exposure, a missing fundus. It is not a gradability label. The literature reports a larger
  share of EyePACS images as hard to grade (about a quarter), and many of those pass this flag.
- **Inverted / flipped images.** Some EyePACS images are mirrored or upside down. These are not
  detected. Rotation and flip augmentation in training makes the model insensitive to them.
- **Resizing.** The 1024 px copy loses the original camera resolution, so resolution can no longer
  identify the camera; aspect ratio still can.

### Quality flag thresholds (`configs/base.yaml` → `quality`)

The statistics are computed inside the fundus mask after resizing the long side to 512 px. The
thresholds were set by viewing real images at different percentiles of each statistic:

| Flag | Rule | Why this value |
|---|---|---|
| `poor:unreadable` | the file does not decode | — |
| `poor:too_dark` | brightness < 20, or a small mask on a dark frame | 20–25 is dark but vessels are visible; below 20 they are not |
| `poor:no_fundus` | fundus fraction < 0.35 on a frame that isn't dark | below 0.35 are black frames and corrupted composites |
| `poor:overexposed` | brightness > 220 | 180–220 are pale but normal fundi |
| `poor:blurred` | sharpness < 60 | below 60 (the bottom ~2%) is almost always hazy; 60–90 is mixed |

Thresholds are re-applied without re-scanning the images, because statistics are cached.

---

## Manifest data dictionary (`data/processed/manifest.csv`)

| Column | Type | Description |
|---|---|---|
| `image` | str | File stem, e.g. `10_left`; the key into the labels CSV |
| `patient_id` | str | Numeric patient id parsed from `image`. Kept as a string so leading digits are never lost. |
| `eye` | str | `left` or `right` |
| `level` | int | ICDR grade 0–4 from the labels CSV |
| `label` | int | 1 if `level` ≥ `data.referable_min_grade` (2), else 0 |
| `path` | str | Image path relative to `paths.raw_dir`, with forward slashes |
| `file_bytes` | int | File size in bytes |
| `md5` | str | MD5 of the file bytes, used to group byte-identical files in the split |
| `readable` | bool | Whether the file decodes as an image |
| `width`, `height` | int | Original pixel size of the file |
| `brightness` | float | Mean grey level inside the fundus mask (0–255, at 512 px) |
| `contrast` | float | Standard deviation of grey level inside the fundus mask |
| `sharpness` | float | Variance of the Laplacian inside the eroded fundus mask (at 512 px) |
| `fundus_fraction` | float | Share of image pixels inside the fundus mask |
| `aspect_ratio` | float | `width / height`; a proxy for camera type |
| `dark_p99` | float | 99th percentile of the max colour channel; separates dark frames from frames with no fundus |
| `quality` | str | `ok` or `poor:<unreadable|too_dark|no_fundus|overexposed|blurred>` |
| `quality_ok` | bool | `quality == "ok"` |

Phase 3 adds `split` (`train`, `val`, `test`) and the group id used to keep patients and identical
files together.
