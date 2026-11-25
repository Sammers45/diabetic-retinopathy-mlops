"""Phase 4: models and training with MLflow tracking (SYNTHETIC data, random-init weights).

Needs TensorFlow. Skipped where it cannot load (see conftest.require_tensorflow); CI runs it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import require_tensorflow
from dr.config import load_config
from dr.train import class_weights

tf = require_tensorflow()
keras = pytest.importorskip("keras")

from dr import data as D  # noqa: E402
from dr import models as M  # noqa: E402
from dr import train as T  # noqa: E402

SIZE = 64


def tiny_overrides(tmp: Path, raw: Path, family: str = "efficientnet_b0", **extra: str) -> dict[str, str]:
    """Tiny, fast settings: 64 px images, 1-2 epochs, small head, sqlite in the temp dir."""
    return {
        "paths.raw_dir": raw.as_posix(),
        "paths.processed_dir": (tmp / "processed").as_posix(),
        "paths.reports_dir": (tmp / "reports").as_posix(),
        "mlflow.tracking_uri": f"sqlite:///{(tmp / 'mlruns.db').as_posix()}",
        "data.max_images": "null",
        "preprocess.image_size": str(SIZE),
        "model.family": family,
        "model.pretrained": "false",
        "model.embedding_dim": "16",
        "model.fine_tune_layers": "5",
        "train.batch_size": "8",
        "train.head_epochs": "1",
        "train.finetune_epochs": "1",
        **extra,
    }


def tiny_cfg(tmp: Path, raw: Path, family: str = "efficientnet_b0", **extra: str):
    overrides = tiny_overrides(tmp, raw, family, **extra)
    return load_config("local_cpu", overrides=[f"{k}={v}" for k, v in overrides.items()])


@pytest.fixture(scope="module")
def prepared(synthetic_data, tmp_path_factory):
    """Synthetic fixtures run through Phase 2-3 (manifest, split, 64 px PNG cache)."""
    tmp = tmp_path_factory.mktemp("p4")
    cfg = tiny_cfg(tmp, synthetic_data.root)
    df, report = D.build_manifest(cfg, workers=1)
    df, _ = D.make_splits(df, cfg)
    D.write_outputs(df, report, tmp / "processed")
    D.cache_preprocessed(df, cfg, workers=1)
    return tmp, synthetic_data.root


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("family", ["efficientnet_b0", "baseline_cnn"])
def test_model_has_embedding_and_output(family: str, tmp_path: Path, synthetic_data) -> None:
    cfg = tiny_cfg(tmp_path, synthetic_data.root, family=family)
    model = M.build_model(cfg, pretrained=False)
    assert model.get_layer(M.EMBEDDING).output.shape[-1] == 16
    x = np.full((2, SIZE, SIZE, 3), 128, np.float32)
    p = model(x, training=False).numpy()
    assert p.shape == (2, 1) and np.all((p >= 0) & (p <= 1))
    emb, score = M.predict_embeddings(model, x)
    assert emb.shape == (2, 16) and np.allclose(score, p.reshape(-1), atol=1e-6)


def test_augmentation_is_active_only_in_training(tmp_path: Path, synthetic_data) -> None:
    """Keras 3 augmentation layers default to training mode when `training` is omitted; make sure
    our inference helpers never hit that."""
    model = M.build_model(tiny_cfg(tmp_path, synthetic_data.root), pretrained=False)
    x = np.random.default_rng(1).uniform(0, 255, (2, SIZE, SIZE, 3)).astype(np.float32)
    aug = model.get_layer("augmentation")
    assert not np.allclose(aug(x, training=True).numpy(), x)
    assert np.allclose(aug(x, training=False).numpy(), x)
    _, s1 = M.predict_embeddings(model, x)
    _, s2 = M.predict_embeddings(model, x)
    assert np.array_equal(s1, s2)
    assert np.allclose(model.predict(x, verbose=0).reshape(-1), s1, atol=1e-6)


def test_inference_is_deterministic_augmentation_off(tmp_path: Path, synthetic_data) -> None:
    model = M.build_model(tiny_cfg(tmp_path, synthetic_data.root), pretrained=False)
    x = np.random.default_rng(0).uniform(0, 255, (3, SIZE, SIZE, 3)).astype(np.float32)
    a = model(x, training=False).numpy()
    b = model(x, training=False).numpy()
    assert np.array_equal(a, b)


def test_finetune_unfreezes_top_layers_but_keeps_batchnorm_frozen(tmp_path: Path, synthetic_data) -> None:
    model = M.build_model(tiny_cfg(tmp_path, synthetic_data.root), pretrained=False)
    M.freeze_backbone(model)
    assert not M.get_backbone(model).trainable
    n = M.unfreeze_top(model, 40)
    backbone = M.get_backbone(model)
    assert 0 < n <= 40
    assert all(not layer.trainable for layer in backbone.layers if isinstance(layer, keras.layers.BatchNormalization))
    assert all(not layer.trainable for layer in backbone.layers[:-40])


def test_pretrained_weights_request_the_real_file_name(monkeypatch) -> None:
    """Regression: naming the application 'backbone' made Keras fetch 'backbone_notop.h5' (403)."""
    requested: list[str] = []

    def fake_get_file(fname=None, origin=None, *a, **kw):
        requested.append(fname or origin)
        raise RuntimeError("stop after recording the request")

    monkeypatch.setattr("keras.src.applications.efficientnet.file_utils.get_file", fake_get_file)
    with pytest.raises(RuntimeError, match="stop after"):
        M.build_backbone("efficientnet_b0", SIZE, pretrained=True)
    assert requested and requested[0].startswith("efficientnetb0"), requested


def test_unknown_family_rejected(tmp_path: Path, synthetic_data) -> None:
    with pytest.raises(ValueError, match="model.family"):
        M.build_model(tiny_cfg(tmp_path, synthetic_data.root, family="resnet9000"))


def test_class_weights_balance_classes() -> None:
    w = class_weights(pd.Series([0] * 80 + [1] * 20))
    assert w[1] / w[0] == pytest.approx(4.0)
    assert 80 * w[0] == pytest.approx(20 * w[1])


# ------------------------------------------------------------------ data pipeline
def test_dataset_reads_cached_pngs_exactly(prepared) -> None:
    tmp, raw = prepared
    cfg = tiny_cfg(tmp, raw)
    df = T.load_split_manifest(cfg)
    ds = T.make_dataset(df.head(4), batch_size=4, shuffle=False, seed=0)
    images, labels = next(iter(ds))
    from dr import preprocess as P

    expected = np.stack([P.load_png(p) for p in df.head(4)["png"]]).astype(np.float32)
    assert np.array_equal(images.numpy(), expected), "TF PNG decode must match OpenCV decode bit for bit"
    assert list(labels.numpy()) == list(df.head(4)["label"].astype(float))


def test_missing_cache_is_actionable(prepared) -> None:
    tmp, raw = prepared
    with pytest.raises(D.DataError, match="dr.data"):
        T.load_split_manifest(tiny_cfg(tmp, raw, **{"preprocess.image_size": "96"}))


# ------------------------------------------------------------------ acceptance: a complete MLflow run
@pytest.mark.slow
def test_short_run_logs_a_complete_mlflow_run(prepared, clean_mlflow_env) -> None:
    import mlflow

    tmp, raw = prepared
    overrides = tiny_overrides(tmp, raw, **{"train.head_epochs": "2", "train.finetune_epochs": "1"})
    args = ["--profile", "local_cpu", "--no-pretrained", "--run-name", "fixture-test"]
    for k, v in overrides.items():
        args += ["--set", f"{k}={v}"]
    code = T.main(args)
    assert code == 0
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("dr-training")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    params, tags = run.data.params, run.data.tags
    assert params["preprocess.image_size"] == str(SIZE) and params["model.pretrained"] == "False"
    assert len(params["manifest_md5"]) == 32
    assert tags["profile"] == "local_cpu" and tags["model_family"] == "efficientnet_b0"
    assert {"git_commit", "git_dirty", "gpu", "tensorflow_version"} <= tags.keys()
    assert tags["data"] == "synthetic"
    # Random init trains end to end: 3 epochs (2 + 1) unless early stopping fired.
    hist = client.get_metric_history(run.info.run_id, "val_auc")
    assert [m.step for m in hist] == list(range(len(hist))) and 1 <= len(hist) <= 3
    assert {"loss", "auc", "val_loss", "learning_rate", "train_seconds", "best_val_auc"} <= run.data.metrics.keys()
    arts = {a.path for a in client.list_artifacts(run.info.run_id, "keras")}
    assert {"keras/model.keras", "keras/history.json"} <= arts
    inputs = run.inputs.dataset_inputs
    assert inputs and inputs[0].dataset.name == "manifest"


def test_two_stage_schedule_has_running_step(prepared, clean_mlflow_env, monkeypatch) -> None:
    """Pretrained schedule (head -> fine-tune) keeps one step counter across both stages.

    ImageNet weights are not downloaded in tests; the 'pretrained' code path is exercised by
    marking a random-init model as pretrained.
    """
    import mlflow

    tmp, raw = prepared
    cfg = tiny_cfg(tmp, raw, **{"mlflow.experiment_training": "dr-training-two-stage", "train.early_stopping_patience": "5"})
    real_build = M.build_model

    def fake_pretrained(cfg_, pretrained=None):
        model = real_build(cfg_, pretrained=False)
        model.pretrained_backbone = True
        return model

    monkeypatch.setattr(M, "build_model", fake_pretrained)
    result = T.train(cfg, run_name="two-stage")
    stages = [h["stage"] for h in result.history]
    assert stages == ["head", "finetune"]
    assert [h["step"] for h in result.history] == [0, 1]
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(result.run_id)
    assert int(run.data.params["finetune_trainable_backbone_layers"]) > 0
