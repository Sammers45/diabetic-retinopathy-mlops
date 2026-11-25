# Data

This directory is **git-ignored** except for this file. Images are never committed, for licence
and size reasons.

| Path | Contents | Created in |
|---|---|---|
| `data/raw/` | Kaggle download (EyePACS labels CSV and images) | Phase 1 |
| `data/processed/manifest.csv` | One row per image: ids, label, quality stats, split | Phases 2–3 |
| `data/processed/data_quality.json` | Validation report (missing files, bad names, quality counts) | Phase 2 |
| `data/processed/images_<size>/` | Preprocessed images, cached as lossless PNG | Phase 3 |

The full dataset card and the manifest data dictionary are added in Phase 2.

Tests never use this directory. They use **synthetic** images from
`scripts/make_synthetic_fixtures.py`.
