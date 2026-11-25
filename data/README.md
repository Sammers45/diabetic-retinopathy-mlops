# Data

This directory is **git-ignored** except for this file. Images are never committed, for licence
and size reasons.

| Path | Contents | Created in |
|---|---|---|
| `data/raw/` | Kaggle download (EyePACS labels CSV and images) | Phase 1 |
| `data/processed/manifest.csv` | One row per image: ids, label, quality stats, split | Phases 2–3 |
| `data/processed/data_quality.json` | Validation report (missing files, bad names, quality counts) | Phase 2 |
| `data/processed/images_<size>/` | Preprocessed images, cached as lossless PNG | Phase 3 |

## Getting the data (`python -m dr.download`)

| `data.source` | What happens |
|---|---|
| `kaggle_dataset` (default) | `kagglehub.dataset_download(data.kaggle_dataset)` straight into `data/raw/kaggle/` |
| `kaggle_competition` | Labels plus the sample archive only. Add `--full-competition` for the ~35 GB archive. You must accept the rules at kaggle.com/competitions/diabetic-retinopathy-detection/rules first. |
| `local` | The data is already in `data/raw/`; this only checks that it's there |

The default dataset is
[`tanlikesmath/diabetic-retinopathy-resized`](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized)
(about 7.8 GB), a resized copy of the 2015 competition training set. Its `trainLabels.csv`
(`image`, `level`) has 35,126 rows, and the grade counts 0–4 are 25,810 / 2,443 / 5,292 / 873 /
708 (19.57% referable). This matches the competition. The pipeline uses `trainLabels.csv` with
`resized_train/resized_train/`. The `*_cropped` variant in the same dataset is not used, because
cropping is done by our own preprocessing.

**Credentials** (only their existence is checked; the token is never printed):
`KAGGLE_API_TOKEN`, `~/.kaggle/access_token` (the token only, no file extension) or
`~/.kaggle/kaggle.json`. Each download writes `DOWNLOAD_INFO.json` next to the data, recording the
source, handle, file count, size and time.

The full dataset card and the manifest data dictionary are added in Phase 2.

Tests never use this directory. They use **synthetic** images from
`scripts/make_synthetic_fixtures.py`.
