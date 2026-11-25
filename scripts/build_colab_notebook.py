"""Build notebooks/colab_train.ipynb from cell sources kept in this script.

Why a generator: notebooks are JSON and diff badly. Keeping the cells here makes reviews easy and
guarantees the notebook only calls ``src/`` code. Re-run after editing:
``python scripts/build_colab_notebook.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "colab_train.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS = [
    md(
        """
# Referable DR: Colab training

Runs the pipeline on the full dataset with a GPU. **Not a medical device.**

Edit only `REPO_URL` (cell 1) and the data source and handle (cell 3). Everything else runs as-is.
Set the runtime to GPU first (Runtime → Change runtime type → T4 GPU).
"""
    ),
    md("## 1. GPU check, clone, install"),
    code(
        """
REPO_URL = "https://github.com/Sammers45/diabetic-retinopathy-mlops.git"

import os, subprocess
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "No GPU found")
if not os.path.exists("/content/repo"):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/repo"], check=True)
%cd /content/repo
%pip install -q -r requirements-colab.txt
"""
    ),
    md("## 2. Drive, secrets, environment"),
    code(
        """
import os
from google.colab import drive, userdata

drive.mount("/content/drive")
os.makedirs("/content/drive/MyDrive/dr-mlops", exist_ok=True)

# Add a Colab secret named KAGGLE_API_TOKEN (key icon in the left sidebar). It is never printed.
try:
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    print("Kaggle token loaded from Colab secrets.")
except Exception as exc:  # secret missing or access not granted
    print(f"KAGGLE_API_TOKEN secret not available ({type(exc).__name__}); download will fail.")

os.environ["DR_PROFILE"] = "colab"
os.environ["PYTHONPATH"] = "src"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
# Optional: use a remote MLflow server instead of the sqlite store on Drive.
# os.environ["MLFLOW_TRACKING_URI"] = "https://your-mlflow-server"

!python -m dr.config | head -n 40
"""
    ),
    md(
        """
## 3. Download data *(Phase 1)*
## 4. Ingest and data quality *(Phase 2)*
## 5. Train, evaluate, register *(Phases 4–5)*
## 6. Monitoring replay *(Phase 7)*
## 7. Viewing runs

These cells are added as each phase lands.
"""
    ),
]


def build() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
