"""Build notebooks/01_eda.ipynb (cells kept here so the notebook stays a thin caller of src/).

The committed notebook has NO outputs: example cells display real EyePACS photographs, which
must not be redistributed in the repository. Run it locally after ``python -m dr.data``.
"""

from __future__ import annotations

import json
from pathlib import Path

from build_colab_notebook import code, md

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"

CELLS = [
    md(
        """
# EyePACS: exploratory data analysis

Thin notebook over `dr.eda` and `dr.data`. Run `python -m dr.data` first, which writes
`data/processed/manifest.csv` and `data_quality.json`.

**Do not commit this notebook with outputs.** The example cells show licensed photographs.
"""
    ),
    code(
        """
import json, sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "dr").is_dir())
sys.path.insert(0, str(ROOT / "src"))

from dr import eda
from dr.config import load_config, resolve_path

cfg = load_config()
RAW = resolve_path(cfg, "paths.raw_dir")
PROCESSED = resolve_path(cfg, "paths.processed_dir")
df = eda.load_manifest(PROCESSED / "manifest.csv")
report = json.loads((PROCESSED / "data_quality.json").read_text())
print(f"{len(df):,} images, {df.patient_id.nunique():,} patients, referable {df.label.mean():.2%}")
print("validation issues:", {k: v for k, v in report["issues"].items() if k.endswith(("count", "rows")) or k == "images_without_label"})
"""
    ),
    md("## Grade distribution"),
    code("eda.plot_grades(df);"),
    md("## Image quality flags\nThresholds are in `configs/base.yaml` → `quality`."),
    code(
        """
print(json.dumps(report["quality_counts"], indent=1), "\\npoor share:", report["poor_quality_share"])
eda.plot_quality(df);
"""
    ),
    md("## Image statistics\nThese are also the image features monitored for drift after deployment."),
    code("eda.plot_stat_histograms(df);"),
    md("## Resolution and camera groups\nThe resized copy fixes the width at 1024 px, but the aspect ratio still identifies the camera type."),
    code(
        """
eda.plot_resolutions(df)
eda.plot_referable_by_group(df);
"""
    ),
    md("## Example images per grade *(display only)*"),
    code("eda.show_examples(df[df.quality_ok], RAW, by='level', n=5, seed=1);"),
    md("## Examples of poor-quality images *(display only)*"),
    code("eda.show_examples(df[~df.quality_ok], RAW, by='quality', n=5, seed=1);"),
    md("## Duplicates\nFiles that share an MD5 are grouped together in the patient-level split (Phase 3)."),
    code("report['md5_duplicates']"),
]


def build() -> dict:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    return nb


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
