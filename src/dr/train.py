"""Train a referable-DR classifier and track it in MLflow (``python -m dr.train``).

Input: ``data/processed/manifest.csv`` (with ``split``) and the PNG cache, both from
``python -m dr.data``. The validation set is always the explicit ``split == "val"`` rows; Keras
``validation_split`` is never used, because it takes the last rows *before* shuffling.

Schedule:

* Pretrained backbone, two stages: (1) head only, backbone frozen; (2) the top
  ``model.fine_tune_layers`` backbone layers unfrozen, BatchNorm still frozen, lower learning
  rate. Early stopping on validation AUC in both stages, best weights restored.
* Random init (``--no-pretrained``, offline tests) or ``baseline_cnn``: a single end-to-end stage.

MLflow (experiment ``mlflow.experiment_training``): the full flattened config as parameters, tags
for profile, git commit, GPU and model family, per-epoch metrics with one running step across
both stages, the manifest as a dataset input (and its MD5 as a parameter), and the trained Keras
model plus training history as artifacts. Phase 5 adds evaluation and registration to the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dr import data as D
from dr.config import ConfigError, add_config_args, config_from_args, flatten, get, resolve_path
from dr.tracking import git_commit, git_dirty, setup_mlflow

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclass
class TrainResult:
    run_id: str
    model: Any
    history: list[dict[str, float]]
    splits: dict[str, pd.DataFrame]
    evaluation: dict[str, Any] | None = None
    model_version: str | None = None
    alias: str | None = None


# --------------------------------------------------------------------------- data
def load_split_manifest(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Manifest rows in train/val/test with their cached PNG path; fails loudly if stale."""
    path = resolve_path(cfg, "paths.processed_dir") / D.MANIFEST
    if not path.exists():
        raise D.DataError(f"{path} not found. Run `python -m dr.data` first.")
    df = pd.read_csv(path, dtype={"patient_id": str, "image": str})
    if "split" not in df.columns:
        raise D.DataError("manifest.csv has no split column. Re-run `python -m dr.data`.")
    df = df[df["split"].isin(D.SPLITS)].copy()
    df["png"] = [str(D.cache_path(cfg, s)) for s in df["image"]]
    missing = [p for p in df["png"] if not Path(p).exists()]
    if missing:
        raise D.DataError(
            f"{len(missing)} cached PNGs missing for image size {get(cfg, 'preprocess.image_size')} "
            f"(e.g. {missing[0]}). Re-run `python -m dr.data` with the same profile/overrides."
        )
    return df


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_dataset(df: pd.DataFrame, batch_size: int, shuffle: bool, seed: int):
    """``tf.data`` pipeline reading cached PNGs (lossless, so identical to serving inputs)."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((df["png"].to_numpy(), df["label"].astype("float32").to_numpy()))
    if shuffle:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        return tf.cast(img, tf.float32), label

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(labels: pd.Series) -> dict[int, float]:
    """Inverse-frequency weights so both classes contribute equally to the loss."""
    n = len(labels)
    counts = labels.value_counts()
    return {int(c): float(n / (2 * counts.get(c, 1))) for c in (0, 1)}


# --------------------------------------------------------------------------- MLflow
class MlflowEpochLogger:
    """Keras callback logging every epoch with one running step across both stages."""

    def __new__(cls, *args, **kwargs):  # build the subclass lazily so importing needs no TF
        import keras

        class _Logger(keras.callbacks.Callback):
            def __init__(self, stage: str, step_offset: int, sink: list[dict[str, float]]) -> None:
                super().__init__()
                self.stage, self.offset, self.sink = stage, step_offset, sink

            def on_epoch_end(self, epoch: int, logs: Mapping[str, float] | None = None) -> None:
                import mlflow

                step = self.offset + len([h for h in self.sink if h["stage"] == self.stage])
                metrics = {k: float(v) for k, v in (logs or {}).items()}
                lr = self.model.optimizer.learning_rate
                metrics["learning_rate"] = float(lr.numpy() if hasattr(lr, "numpy") else lr)
                mlflow.log_metrics(metrics, step=step)
                self.sink.append({"stage": self.stage, "step": step, **metrics})

        return _Logger(*args, **kwargs)


def log_config(cfg: Mapping[str, Any]) -> None:
    """Every config value as a run parameter (MLflow accepts at most 100 per call)."""
    import mlflow

    flat = {k: ("null" if v is None else str(v))[:6000] for k, v in flatten(cfg).items()}
    items = list(flat.items())
    for i in range(0, len(items), 100):
        mlflow.log_params(dict(items[i : i + 100]))


def gpu_names() -> list[str]:
    import tensorflow as tf

    return [d.name for d in tf.config.list_physical_devices("GPU")]


# --------------------------------------------------------------------------- training
def _compile(model, lr: float) -> None:
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")],
    )


def _fit(model, train_ds, val_ds, epochs: int, patience: int, weights, stage: str, offset: int, sink) -> int:
    """Run one stage; returns the number of epochs actually run."""
    import keras

    if epochs <= 0:
        return 0
    before = len(sink)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=weights,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience, restore_best_weights=True),
            MlflowEpochLogger(stage, offset, sink),
        ],
    )
    return len(sink) - before


def train(cfg: Mapping[str, Any], pretrained: bool | None = None, run_name: str | None = None) -> TrainResult:
    """Train with MLflow tracking and return the run id, the model and the split tables."""
    import keras
    import mlflow
    import tensorflow as tf

    from dr import models as M

    seed = int(get(cfg, "project.seed", 42))
    keras.utils.set_random_seed(seed)
    gpus = gpu_names()
    if get(cfg, "train.mixed_precision", False) and gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        keras.mixed_precision.set_global_policy("float32")

    df = load_split_manifest(cfg)
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in D.SPLITS}
    bs = int(get(cfg, "train.batch_size", 32))
    train_ds = make_dataset(splits["train"], bs, shuffle=True, seed=seed)
    val_ds = make_dataset(splits["val"], bs, shuffle=False, seed=seed)
    weights = class_weights(splits["train"]["label"]) if get(cfg, "train.class_weights", True) else None
    patience = int(get(cfg, "train.early_stopping_patience", 3))

    model = M.build_model(cfg, pretrained=pretrained)
    family = str(get(cfg, "model.family"))
    manifest_path = resolve_path(cfg, "paths.processed_dir") / D.MANIFEST
    info = setup_mlflow(cfg, str(get(cfg, "mlflow.experiment_training")))
    history: list[dict[str, float]] = []

    with mlflow.start_run(run_name=run_name or f"{family}-{get(cfg, 'meta.profile')}") as run:
        log_config(cfg)
        mlflow.log_params(
            {
                "manifest_md5": file_md5(manifest_path),
                "pretrained_backbone": model.pretrained_backbone,
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(splits["test"]),
                "class_weight_pos": None if weights is None else round(weights[1], 4),
                "total_params": model.count_params(),
            }
        )
        mlflow.set_tags(
            {
                "profile": get(cfg, "meta.profile"),
                "git_commit": git_commit() or "unknown",
                "git_dirty": str(git_dirty()).lower(),
                "gpu": ",".join(gpus) or "none",
                "model_family": family,
                "tensorflow_version": tf.__version__,
                "phase": "training",
                "data": "synthetic" if "synthetic" in str(resolve_path(cfg, "paths.raw_dir")).lower() else "real",
            }
        )
        cols = ["image", "patient_id", "eye", "level", "label", "split", "quality", "group_id"]
        mlflow.log_input(
            mlflow.data.from_pandas(df[cols], source=str(manifest_path), targets="label", name="manifest"),
            context="training",
        )

        t0 = time.time()
        if model.pretrained_backbone:
            M.freeze_backbone(model)
            _compile(model, float(get(cfg, "train.head_lr")))
            n1 = _fit(model, train_ds, val_ds, int(get(cfg, "train.head_epochs")), patience, weights, "head", 0, history)
            n_trainable = M.unfreeze_top(model, int(get(cfg, "model.fine_tune_layers")))
            mlflow.log_param("finetune_trainable_backbone_layers", n_trainable)
            _compile(model, float(get(cfg, "train.finetune_lr")))
            _fit(model, train_ds, val_ds, int(get(cfg, "train.finetune_epochs")), patience, weights, "finetune", n1, history)
        else:
            epochs = int(get(cfg, "train.head_epochs")) + int(get(cfg, "train.finetune_epochs"))
            _compile(model, float(get(cfg, "train.head_lr")))
            _fit(model, train_ds, val_ds, epochs, patience, weights, "end_to_end", 0, history)
        mlflow.log_metric("train_seconds", time.time() - t0)
        if history:
            best = max(history, key=lambda h: h.get("val_auc", -1))
            mlflow.log_metrics({"best_val_auc": best["val_auc"], "best_step": best["step"]})

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.keras"
            model.save(model_path)
            (Path(tmp) / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(model_path), artifact_path="keras")
            mlflow.log_artifact(str(Path(tmp) / "history.json"), artifact_path="keras")
        result = TrainResult(run.info.run_id, model, history, splits)
        if get(cfg, "evaluate.enabled", True):
            evaluate_and_register(cfg, result, bs)
        print(f"MLflow run {run.info.run_id} in experiment {info.experiment_name} ({info.tracking_uri})")
        return result


def score_split(model, df: pd.DataFrame, batch_size: int) -> np.ndarray:
    """Inference-mode scores for one split (``predict`` never applies augmentation)."""
    return model.predict(make_dataset(df, batch_size, shuffle=False, seed=0), verbose=0).reshape(-1)


def evaluate_and_register(cfg: Mapping[str, Any], result: TrainResult, batch_size: int) -> None:
    """Phase 5 inside the training run: threshold on val, test once, reference profile, registry."""
    import mlflow

    from dr import evaluate as E
    from dr import reference as R
    from dr import serving_model as S

    scored = {}
    for s in ("val", "test"):
        part = result.splits[s].copy()
        part["p_referable"] = score_split(result.model, part, batch_size)
        scored[s] = part
    with tempfile.TemporaryDirectory() as tmp:
        eval_dir, ref_dir = Path(tmp) / "evaluation", Path(tmp) / "reference_profile"
        summary = E.evaluate(scored["val"], scored["test"], cfg, eval_dir)
        mlflow.log_metrics(E.mlflow_metrics(summary))
        mlflow.log_artifacts(str(eval_dir), artifact_path="evaluation")
        pca, ref = R.build_reference(result.model, result.splits["train"], cfg, summary["threshold"], ref_dir)
        mlflow.log_artifacts(str(ref_dir), artifact_path="reference_profile")
        mlflow.log_metrics(
            {"reference_positive_rate": ref["positive_rate"], "reference_pca_explained_variance": ref["pca_explained_variance"]}
        )
    run_tags = mlflow.tracking.MlflowClient().get_run(result.run_id).data.tags
    tags = {k: run_tags.get(k, "") for k in ("profile", "data", "model_family")}
    mv, alias, _ = S.log_and_register(result.model, pca, cfg, summary["threshold"], result.run_id, extra_tags=tags)
    mlflow.set_tags(
        {"registered_model": str(get(cfg, "mlflow.registered_model")), "registered_version": str(mv.version), "registered_alias": alias}
    )
    result.evaluation, result.model_version, result.alias = summary, str(mv.version), alias
    t = summary["test"]
    ci = summary["test_ci95"]
    print(
        f"Threshold {summary['threshold']:.4f} (val sensitivity {summary['val']['sensitivity']:.3f}). "
        f"TEST: AUC {t['roc_auc']:.3f} [{ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}], "
        f"sens {t['sensitivity']:.3f}, spec {t['specificity']:.3f}. Registered "
        f"{get(cfg, 'mlflow.registered_model')} v{mv.version} @{alias}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(argparse.ArgumentParser(description="Train and track a referable-DR model."))
    parser.add_argument("--no-pretrained", action="store_true", help="Random init (offline tests; trains end to end).")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
        if args.no_pretrained:
            cfg["model"]["pretrained"] = False
        result = train(cfg, pretrained=False if args.no_pretrained else None, run_name=args.run_name)
    except (D.DataError, ConfigError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 2
    print(f"Done: run_id={result.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
