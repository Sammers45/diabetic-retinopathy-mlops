"""Phase 5: threshold, metrics, reference profile, pyfunc model and registry (SYNTHETIC data)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dr import evaluate as E

# ------------------------------------------------------------------ pure metric logic (no TF)


def test_threshold_is_highest_with_target_sensitivity() -> None:
    y = np.array([1] * 10 + [0] * 10)
    p = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2] + [0.1] * 5 + [0.45, 0.55, 0.65, 0.75, 0.88])
    t = E.choose_threshold(y, p, 0.90)
    assert t == pytest.approx(0.3)  # 9 of 10 positives are >= 0.3
    assert E.binary_metrics(y, p, t)["sensitivity"] >= 0.90
    assert E.binary_metrics(y, p, np.nextafter(t, 1))["sensitivity"] < 0.90, "no higher threshold qualifies"
    with pytest.raises(ValueError):
        E.choose_threshold(np.zeros(5), np.linspace(0, 1, 5))


def test_binary_metrics_known_values() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.3, 0.2, 0.1, 0.05])
    m = E.binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["ppv"] == pytest.approx(2 / 3) and m["npv"] == pytest.approx(4 / 5)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert 0.8 < m["roc_auc"] <= 1.0


def test_bootstrap_is_patient_clustered_and_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"patient_id": np.repeat(np.arange(n // 2), 2).astype(str), "label": rng.integers(0, 2, n)})
    df["p_referable"] = np.clip(df["label"] * 0.4 + rng.uniform(0, 0.6, n), 0, 1)
    ci = E.bootstrap_ci(df, 0.5, n_boot=200, seed=1)
    point = E.binary_metrics(df["label"], df["p_referable"], 0.5)
    for k in ("roc_auc", "sensitivity", "specificity"):
        lo, hi = ci[k]
        assert lo <= point[k] <= hi and hi - lo < 0.3
    assert ci == E.bootstrap_ci(df, 0.5, n_boot=200, seed=1), "deterministic for a fixed seed"


def test_evaluate_writes_artifacts(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)

    def part(n: int) -> pd.DataFrame:
        y = rng.integers(0, 2, n)
        return pd.DataFrame(
            {
                "image": [f"{i}_left" for i in range(n)],
                "patient_id": [str(i) for i in range(n)],
                "eye": rng.choice(["left", "right"], n),
                "label": y,
                "quality": rng.choice(["ok", "poor:blurred"], n, p=[0.9, 0.1]),
                "aspect_ratio": rng.choice([1.33, 1.5], n),
                "p_referable": np.clip(y * 0.5 + rng.uniform(0, 0.5, n), 0, 1),
            }
        )

    cfg = {"evaluate": {"target_sensitivity": 0.9, "n_bootstrap": 100}, "project": {"seed": 0}}
    summary = E.evaluate(part(200), part(200), cfg, tmp_path)
    for f in ("metrics.json", "test_predictions.csv", "subgroup_metrics.csv", "roc_test.png", "calibration_test.png", "confusion_test.png"):
        assert (tmp_path / f).exists(), f
    assert summary["val"]["sensitivity"] >= 0.9
    preds = pd.read_csv(tmp_path / "test_predictions.csv")
    assert {"p_referable", "prediction", "threshold"} <= set(preds.columns)
    sub = pd.read_csv(tmp_path / "subgroup_metrics.csv")
    assert set(sub["subgroup"]) == {"eye", "camera_aspect", "quality"}
    flat = E.mlflow_metrics(summary)
    assert {"threshold", "test_roc_auc", "test_roc_auc_ci_low", "test_sensitivity_ci_high"} <= flat.keys()


# ------------------------------------------------------------------ end to end with TensorFlow + MLflow
# The `champion` fixture (a tiny model trained on synthetic data, registered as @champion) lives
# in conftest.py so Phase 5 and Phase 6 share one training run.


def _load_champion(cfg):
    import mlflow

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    return mlflow.pyfunc.load_model("models:/dr-referable@champion")


def test_run_has_evaluation_reference_and_registration(champion) -> None:
    import mlflow

    result, cfg, _, _ = champion
    assert result.alias == "champion" and result.model_version == "1"
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(result.run_id)
    assert {"threshold", "test_roc_auc", "test_sensitivity", "test_brier", "test_roc_auc_ci_low"} <= run.data.metrics.keys()
    assert run.data.tags["registered_alias"] == "champion"
    eval_files = {Path(a.path).name for a in client.list_artifacts(result.run_id, "evaluation")}
    assert {"roc_test.png", "calibration_test.png", "confusion_test.png", "test_predictions.csv", "metrics.json"} <= eval_files
    ref_files = {Path(a.path).name for a in client.list_artifacts(result.run_id, "reference_profile")}
    assert {"reference.parquet", "pca.joblib", "summary.json"} <= ref_files
    mv = client.get_model_version_by_alias("dr-referable", "champion")
    assert mv.tags["training_run_id"] == result.run_id and mv.tags["data"] == "synthetic"


def test_serving_predictions_equal_training_time_test_predictions(champion) -> None:
    """Acceptance: models:/dr-referable@champion reproduces the logged test predictions (atol 1e-5)."""
    import mlflow

    result, cfg, tmp, fixtures = champion
    local = mlflow.artifacts.download_artifacts(run_id=result.run_id, artifact_path="evaluation/test_predictions.csv", dst_path=str(tmp / "dl"))
    logged = pd.read_csv(local)
    manifest = pd.read_csv(tmp / "processed" / "manifest.csv", dtype={"image": str}).set_index("image")
    frame = pd.DataFrame({"image_path": [str(fixtures.root / manifest.loc[i, "path"]) for i in logged["image"]]})
    out = _load_champion(cfg).predict(frame)
    assert out["error"].isna().all()
    np.testing.assert_allclose(out["p_referable"].to_numpy(), logged["p_referable"].to_numpy(), atol=1e-5)
    assert (out["prediction"].astype(int).to_numpy() == logged["prediction"].to_numpy()).all()
    assert out["threshold"].nunique() == 1 and out["threshold"].iloc[0] == pytest.approx(logged["threshold"].iloc[0])
    emb_cols = [c for c in out.columns if c.startswith("emb_")]
    assert len(emb_cols) == 8 and out[emb_cols].notna().all().all()
    assert {"width", "height", "brightness", "contrast", "sharpness", "fundus_fraction", "quality"} <= set(out.columns)


def test_base64_input_matches_path_input(champion) -> None:
    _, cfg, tmp, fixtures = champion
    manifest = pd.read_csv(tmp / "processed" / "manifest.csv", dtype={"image": str})
    paths = [str(fixtures.root / p) for p in manifest[manifest["quality"] == "ok"]["path"].head(3)]
    b64 = [base64.b64encode(Path(p).read_bytes()).decode() for p in paths]
    model = _load_champion(cfg)
    by_path = model.predict(pd.DataFrame({"image_path": paths}))
    by_b64 = model.predict(pd.DataFrame({"image_b64": b64}))
    data_uri = model.predict(pd.DataFrame({"image_b64": ["data:image/jpeg;base64," + b64[0]]}))
    np.testing.assert_allclose(by_b64["p_referable"], by_path["p_referable"], atol=1e-6)
    assert data_uri["p_referable"].iloc[0] == pytest.approx(by_path["p_referable"].iloc[0], abs=1e-6)


def test_corrupt_upload_does_not_crash_the_batch(champion) -> None:
    _, cfg, tmp, fixtures = champion
    good = str(next((fixtures.image_dir).glob("*_left.jpeg")))
    frame = pd.DataFrame(
        {
            "image_path": [good, str(fixtures.image_dir / f"{fixtures.unreadable}.jpeg"), None, str(tmp / "missing.jpeg"), None],
            "image_b64": [None, None, base64.b64encode(b"not an image").decode(), None, "%%%not-base64%%%"],
        }
    )
    out = _load_champion(cfg).predict(frame)
    assert len(out) == 5
    assert out["error"].iloc[0] is None or pd.isna(out["error"].iloc[0])
    assert 0 <= out["p_referable"].iloc[0] <= 1
    for i in range(1, 5):
        assert out["quality"].iloc[i] == "poor:unreadable", out.iloc[i].to_dict()
        assert np.isnan(out["p_referable"].iloc[i]) and isinstance(out["error"].iloc[i], str)


def test_second_registration_becomes_challenger(champion) -> None:
    import mlflow

    from dr import serving_model as S

    result, cfg, _, _ = champion
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    pca = __import__("joblib").load(mlflow.artifacts.download_artifacts(run_id=result.run_id, artifact_path="reference_profile/pca.joblib"))
    with mlflow.start_run(run_name="second"):
        mv, alias, _ = S.log_and_register(result.model, pca, cfg, 0.5, result.run_id, {"data": "synthetic"})
    assert alias == "challenger" and str(mv.version) == "2"
    client = mlflow.tracking.MlflowClient()
    assert str(client.get_model_version_by_alias("dr-referable", "champion").version) == "1"
    meta = json.loads(Path(mlflow.artifacts.download_artifacts(f"models:/dr-referable/2", dst_path=None)).joinpath("artifacts", "serving_config.json").read_text())
    assert meta["threshold"] == 0.5 and meta["n_emb"] == 8
