"""The registered model: one MLflow pyfunc holding everything needed to serve and monitor.

Contents: the Keras network, the preprocessing settings (applied with ``dr.preprocess``, the
same code that built the training cache), the operating threshold, class names and the fitted
PCA used for drift features.

Input: a DataFrame with ``image_path`` and/or ``image_b64`` (both optional string columns).
Output, one row per input row:

``p_referable, prediction, threshold, quality, width, height, brightness, contrast, sharpness,
fundus_fraction, emb_0 .. emb_{k-1}, error``

A bad input (missing, unreadable, not an image, invalid base64) never fails the batch: its row
gets ``quality = "poor:unreadable"``, NaN scores and an ``error`` message.
"""

from __future__ import annotations

import base64
import binascii
import json
import tempfile
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dr import preprocess as P

KERAS_FILE = "model.keras"
PCA_FILE = "pca.joblib"
CONFIG_FILE = "serving_config.json"
CLASS_NAMES = ["not_referable", "referable"]
UNREADABLE = "poor:unreadable"


# --------------------------------------------------------------------------- input decoding
def _decode_b64(text: str) -> bytes:
    """Base64 (optionally a ``data:image/...;base64,`` URI) -> bytes."""
    if "," in text[:100] and text.lstrip().startswith("data:"):
        text = text.split(",", 1)[1]
    return base64.b64decode("".join(text.split()), validate=True)


def _row_bytes(row: Mapping[str, Any]) -> bytes:
    b64 = row.get("image_b64")
    path = row.get("image_path")
    if isinstance(b64, str) and b64.strip():
        try:
            return _decode_b64(b64)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"invalid base64: {exc}") from exc
    if isinstance(path, str) and path.strip():
        try:
            return Path(path).read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read image_path: {exc.strerror or exc}") from exc
    raise ValueError("row has neither image_path nor image_b64")


# --------------------------------------------------------------------------- prediction
def predict_frame(keras_model, pca, meta: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of inputs. Pure function so it can be tested without MLflow."""
    from dr.models import predict_embeddings

    size = int(meta["image_size"])
    pre = meta["preprocess"]
    qcfg = meta["quality"]
    threshold = float(meta["threshold"])
    k = int(meta["n_emb"])
    rows: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    valid: list[int] = []
    for i, rec in enumerate(frame.to_dict("records")):
        out: dict[str, Any] = {"error": None}
        try:
            data = _row_bytes(rec)
            analysis = P.analyse_bytes(data, qcfg)
            out.update({c: analysis[c] for c in P.STAT_COLUMNS})
            out["quality"] = analysis["quality"]
            if not analysis["readable"]:
                out["error"] = "not a readable image"
            else:
                img = P.decode_image(data)
                images.append(P.preprocess_image(img, size, pre, float(qcfg.get("mask_threshold", 10.0))))
                valid.append(i)
        except Exception as exc:  # one bad row must never fail the batch
            out.update({c: np.nan for c in P.STAT_COLUMNS})
            out["quality"] = UNREADABLE
            out["error"] = str(exc)[:300]
        rows.append(out)

    scores = np.full(len(rows), np.nan)
    embs = np.full((len(rows), k), np.nan)
    if valid:
        e, s = predict_embeddings(keras_model, np.stack(images))
        scores[valid] = s
        embs[valid] = pca.transform(e)

    result = pd.DataFrame(rows)
    result.insert(0, "p_referable", scores)
    result.insert(1, "prediction", pd.array(np.where(np.isnan(scores), pd.NA, (scores >= threshold).astype(int)), dtype="Int64"))
    result.insert(2, "threshold", threshold)
    for j in range(k):
        result[f"emb_{j}"] = embs[:, j]
    cols = ["p_referable", "prediction", "threshold", "quality", *P.STAT_COLUMNS, *[f"emb_{j}" for j in range(k)], "error"]
    return result[cols]


try:
    import mlflow.pyfunc

    class DRReferableModel(mlflow.pyfunc.PythonModel):
        """MLflow wrapper; all logic lives in ``predict_frame``."""

        def load_context(self, context) -> None:
            import joblib
            import keras

            self.keras_model = keras.saving.load_model(context.artifacts["keras_model"], compile=False)
            self.pca = joblib.load(context.artifacts["pca"])
            self.meta = json.loads(Path(context.artifacts["serving_config"]).read_text(encoding="utf-8"))

        def predict(self, context, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
            frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
            return predict_frame(self.keras_model, self.pca, self.meta, frame)

except ImportError:  # pragma: no cover - mlflow is a hard requirement in practice
    DRReferableModel = None  # type: ignore[assignment,misc]


# --------------------------------------------------------------------------- logging / registry
def serving_meta(cfg: Mapping[str, Any], threshold: float, n_emb: int, run_id: str) -> dict[str, Any]:
    return {
        "image_size": int(cfg["preprocess"]["image_size"]),
        "preprocess": dict(cfg["preprocess"]),
        "quality": dict(cfg["quality"]),
        "threshold": float(threshold),
        "class_names": CLASS_NAMES,
        "n_emb": int(n_emb),
        "model_family": cfg["model"]["family"],
        "training_run_id": run_id,
        "preprocess_signature": P.preprocess_signature(
            int(cfg["preprocess"]["image_size"]), cfg["preprocess"], float(cfg["quality"].get("mask_threshold", 10.0))
        ),
    }


def signature():
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema

    def out_cols(k: int) -> list:
        cols = [ColSpec("double", "p_referable"), ColSpec("long", "prediction"), ColSpec("double", "threshold"), ColSpec("string", "quality")]
        cols += [ColSpec("double", c) for c in P.STAT_COLUMNS]
        cols += [ColSpec("double", f"emb_{j}") for j in range(k)]
        return [*cols, ColSpec("string", "error")]

    inputs = Schema([ColSpec("string", "image_path", required=False), ColSpec("string", "image_b64", required=False)])
    return ModelSignature(inputs=inputs), out_cols


def _requirements() -> list[str]:
    reqs = []
    for pkg in ("mlflow", "tensorflow", "keras", "scikit-learn", "opencv-python-headless", "numpy", "pandas", "joblib", "pyyaml"):
        try:
            reqs.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            reqs.append(pkg)
    return reqs


def log_and_register(keras_model, pca, cfg: Mapping[str, Any], threshold: float, run_id: str, extra_tags: Mapping[str, str] | None = None):
    """Log the pyfunc into the active run and register it.

    The first registered version becomes ``@champion``; later versions become ``@challenger``
    (promotion is a separate, evidence-based step in Phase 8).
    """
    import joblib
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.types import Schema

    name = str(cfg["mlflow"]["registered_model"])
    k = int(pca.n_components_)
    meta = serving_meta(cfg, threshold, k, run_id)
    in_sig, out_cols = signature()
    sig = ModelSignature(inputs=in_sig.inputs, outputs=Schema(out_cols(k)))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        keras_model.save(tmp_path / KERAS_FILE)
        joblib.dump(pca, tmp_path / PCA_FILE)
        (tmp_path / CONFIG_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=DRReferableModel(),
            artifacts={
                "keras_model": str(tmp_path / KERAS_FILE),
                "pca": str(tmp_path / PCA_FILE),
                "serving_config": str(tmp_path / CONFIG_FILE),
            },
            code_paths=[str(Path(__file__).resolve().parent)],
            signature=sig,
            pip_requirements=_requirements(),
        )
    client = mlflow.tracking.MlflowClient()
    try:
        client.get_model_version_by_alias(name, "champion")
        alias = "challenger"
    except Exception:  # no registered model or no champion yet
        alias = "champion"
    tags = {"training_run_id": run_id, "threshold": f"{threshold:.6f}", **(extra_tags or {})}
    mv = mlflow.register_model(info.model_uri, name, tags=tags)
    client.set_registered_model_alias(name, alias, mv.version)
    return mv, alias, info.model_uri
