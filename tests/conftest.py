"""Shared fixtures. All image data used in tests is SYNTHETIC (see scripts/make_synthetic_fixtures.py)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting in tests/CI


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory: pytest.TempPathFactory):
    """A session-wide synthetic fixture set (generated once, never committed)."""
    from make_synthetic_fixtures import generate_fixtures

    return generate_fixtures(tmp_path_factory.mktemp("synthetic"), n_patients=30, seed=0)


@pytest.fixture
def clean_mlflow_env() -> Iterator[None]:
    """Isolate tests from MLFLOW_TRACKING_URI.

    ``mlflow.set_tracking_uri`` writes the variable into ``os.environ``; without restoring it, one
    test would leak its tracking store into every later test.
    """
    saved = os.environ.pop("MLFLOW_TRACKING_URI", None)
    try:
        yield
    finally:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        if saved is not None:
            os.environ["MLFLOW_TRACKING_URI"] = saved


def require_tensorflow():
    """Import TensorFlow, or skip the test where it cannot load.

    Locally, Windows Smart App Control can block TensorFlow's DLLs; those tests are then skipped.
    CI sets ``DR_REQUIRE_TF=1`` so a broken TensorFlow install fails loudly instead of skipping.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        if os.environ.get("DR_REQUIRE_TF") == "1":
            raise
        pytest.skip(f"TensorFlow unavailable here: {exc}", allow_module_level=True)
    return tf


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def champion(synthetic_data, tmp_path_factory):
    """Train a tiny model on SYNTHETIC data; it is evaluated and registered as @champion.

    Session-scoped because training is the slow part. Tests may register further versions
    (they become @challenger) but must never move @champion.
    """
    require_tensorflow()
    import os

    import mlflow

    from dr import data as D
    from dr import train as T
    from dr.config import load_config

    saved = os.environ.pop("MLFLOW_TRACKING_URI", None)
    tmp = tmp_path_factory.mktemp("p5")
    overrides = {
        "paths.raw_dir": synthetic_data.root.as_posix(),
        "paths.processed_dir": (tmp / "processed").as_posix(),
        "paths.reports_dir": (tmp / "reports").as_posix(),
        "mlflow.tracking_uri": f"sqlite:///{(tmp / 'mlruns.db').as_posix()}",
        "data.max_images": "null",
        "preprocess.image_size": "64",
        "model.pretrained": "false",
        "model.embedding_dim": "16",
        "train.batch_size": "8",
        "train.head_epochs": "1",
        "train.finetune_epochs": "0",
        "evaluate.n_bootstrap": "50",
        "reference.n_images": "40",
        "reference.pca_components": "8",
    }
    cfg = load_config("local_cpu", overrides=[f"{k}={v}" for k, v in overrides.items()])
    df, report = D.build_manifest(cfg, workers=1)
    df, _ = D.make_splits(df, cfg)
    D.write_outputs(df, report, tmp / "processed")
    D.cache_preprocessed(df, cfg, workers=1)
    result = T.train(cfg, pretrained=False, run_name="p5-fixture")
    yield result, cfg, tmp, synthetic_data
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    if saved is not None:
        os.environ["MLFLOW_TRACKING_URI"] = saved
    mlflow.set_tracking_uri(None)
