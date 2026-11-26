"""Phase 6: FastAPI serving and the prediction log (SYNTHETIC data only)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dr.prediction_log import PredictionLog

# ------------------------------------------------------------------ prediction log (no TF)


def test_log_append_adds_embedding_columns_and_reads_windows(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.db")
    log.append([{"request_id": "a", "timestamp": "2026-01-01T00:00:00+00:00", "p_referable": 0.2, "prediction": 0, "emb_0": 1.0, "emb_1": 2.0}])
    log.append([{"request_id": "b", "timestamp": "2026-01-02T00:00:00+00:00", "p_referable": np.float32(0.9), "prediction": np.int64(1), "emb_0": 3.0, "emb_1": 4.0, "emb_2": 5.0}])
    log.append([{"request_id": "c", "p_referable": float("nan"), "prediction": pd.NA, "quality": "poor:unreadable"}])
    assert len(log) == 3 and {"emb_0", "emb_1", "emb_2"} <= set(log.columns())
    df = log.read()
    assert list(df["request_id"]) == ["a", "b", "c"]  # oldest first; "c" got a current timestamp
    assert pd.isna(df.loc[0, "emb_2"]) and df.loc[1, "emb_2"] == 5.0
    assert pd.isna(df.loc[2, "p_referable"]) and pd.isna(df.loc[2, "prediction"])
    window = log.read(start="2026-01-01T12:00:00", end="2026-01-03")
    assert list(window["request_id"]) == ["b"]


def test_log_feedback_sets_label_and_can_be_corrected(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.db")
    log.append([{"request_id": "r1", "p_referable": 0.7, "prediction": 1}])
    row = log.add_feedback("r1", 3)
    assert row is not None and row["grade"] == 3 and row["label"] == 1 and row["feedback_at"]
    row = log.add_feedback("r1", 1)
    assert row["grade"] == 1 and row["label"] == 0
    assert log.add_feedback("missing", 2) is None


def test_log_rejects_unknown_columns(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.db")
    with pytest.raises(ValueError, match="unknown prediction-log column"):
        log.append([{"request_id": "x", "drop table": 1}])


def test_missing_model_gives_actionable_error(tmp_path: Path, clean_mlflow_env) -> None:
    import mlflow

    from dr.api import ModelNotAvailable, load_served_model
    from dr.config import load_config

    cfg = load_config("local_cpu", overrides=[f"mlflow.tracking_uri=sqlite:///{(tmp_path / 'empty.db').as_posix()}"])
    try:
        with pytest.raises(ModelNotAvailable, match="dr-referable@champion"):
            load_served_model(cfg)
    finally:
        mlflow.set_tracking_uri(None)


# ------------------------------------------------------------------ API with the fixture-trained champion
@pytest.fixture(scope="module")
def api(champion, tmp_path_factory):
    from fastapi.testclient import TestClient

    from dr.api import create_app

    result, cfg, tmp, fixtures = champion
    cfg = copy.deepcopy(cfg)
    work = tmp_path_factory.mktemp("p6")
    cfg["serving"]["prediction_log"] = (work / "predictions.db").as_posix()
    cfg["serving"]["image_log_dir"] = (work / "images").as_posix()
    app = create_app(cfg)
    manifest = pd.read_csv(tmp / "processed" / "manifest.csv", dtype={"image": str})
    ok_paths = [fixtures.root / p for p in manifest[manifest["quality"] == "ok"]["path"].head(4)]
    with TestClient(app) as client:
        yield client, app, cfg, result, ok_paths, fixtures, work


def _upload(client, path: Path, content_type: str = "image/jpeg", **form):
    return client.post("/predict", files={"file": (path.name, path.read_bytes(), content_type)}, data=form)


def test_predict_matches_pyfunc_output(api) -> None:
    """Acceptance: /predict returns exactly what the registered pyfunc returns."""
    client, app, _, _, ok_paths, _, _ = api
    expected = app.state.served.pyfunc.predict(pd.DataFrame({"image_path": [str(p) for p in ok_paths]}))
    for path, (_, exp) in zip(ok_paths, expected.iterrows()):
        r = _upload(client, path)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["p_referable"] == pytest.approx(float(exp["p_referable"]), abs=1e-6)
        assert body["prediction"] == int(exp["prediction"])
        assert body["label"] == ("referable" if body["prediction"] else "not_referable")
        assert body["threshold"] == pytest.approx(float(exp["threshold"]))
        assert body["quality"] == exp["quality"]
        assert body["model"] == {"name": "dr-referable", "version": "1", "alias": "champion"}
        assert len(body["request_id"]) == 32 and "Not a medical device" in body["disclaimer"]


def test_predict_appends_full_row_to_log(api) -> None:
    client, app, _, _, ok_paths, _, work = api
    r = _upload(client, ok_paths[0], patient_id="p-17", eye="left")
    rid = r.json()["request_id"]
    row = app.state.log.get(rid)
    assert row is not None
    assert row["source"] == "api" and row["model_version"] == "1" and row["patient_id"] == "p-17" and row["eye"] == "left"
    assert row["p_referable"] == pytest.approx(r.json()["p_referable"])
    assert len(row["md5"]) == 32 and row["filename"] == ok_paths[0].name
    for c in ("width", "height", "brightness", "contrast", "sharpness", "fundus_fraction"):
        assert row[c] is not None
    emb = [k for k in row if k.startswith("emb_")]
    assert len(emb) == 8 and all(row[k] is not None for k in emb)
    assert row["grade"] is None and row["label"] is None
    # log_images defaults to false: nothing written, no path recorded
    assert row["image_path"] is None and not (work / "images").exists()


def test_feedback_updates_the_log(api) -> None:
    client, app, _, _, ok_paths, _, _ = api
    rid = _upload(client, ok_paths[1]).json()["request_id"]
    r = client.post("/feedback", json={"request_id": rid, "grade": 3})
    assert r.status_code == 200, r.text
    assert r.json()["label"] == 1 and r.json()["grade"] == 3
    assert app.state.log.get(rid)["label"] == 1
    r = client.post("/feedback", json={"request_id": rid, "grade": 0})  # correction
    assert r.json()["label"] == 0 and app.state.log.get(rid)["grade"] == 0


def test_feedback_errors(api) -> None:
    client = api[0]
    assert client.post("/feedback", json={"request_id": "nope", "grade": 2}).status_code == 404
    assert client.post("/feedback", json={"request_id": "x", "grade": 5}).status_code == 422
    assert client.post("/feedback", json={"request_id": "x", "grade": -1}).status_code == 422
    assert client.post("/feedback", json={"grade": 2}).status_code == 422


def test_predict_rejects_bad_uploads_cleanly(api) -> None:
    client, app, _, _, ok_paths, fixtures, _ = api
    n_before = len(app.state.log)
    assert _upload(client, ok_paths[0], content_type="text/plain").status_code == 415
    assert client.post("/predict", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}).status_code == 415
    assert client.post("/predict", files={"file": ("e.jpeg", b"", "image/jpeg")}).status_code == 400
    assert client.post("/predict", data={"eye": "left"}).status_code == 422  # no file
    assert _upload(client, ok_paths[0], eye="middle").status_code == 422
    assert len(app.state.log) == n_before  # rejected before scoring: nothing logged

    # Claims to be a JPEG but is not decodable: scored as unreadable, logged, answered with 422.
    r = client.post("/predict", files={"file": ("bad.jpeg", b"\xff\xd8 not really a jpeg", "image/jpeg")})
    assert r.status_code == 422 and r.json()["quality"] == "poor:unreadable"
    row = app.state.log.get(r.json()["request_id"])
    assert row["quality"] == "poor:unreadable" and row["p_referable"] is None and row["error"]
    r = _upload(client, fixtures.image_dir / f"{fixtures.unreadable}.jpeg")
    assert r.status_code == 422
    assert client.get("/health").status_code == 200  # the service survived all of it


def test_oversized_upload_is_rejected(api) -> None:
    from fastapi.testclient import TestClient

    from dr.api import create_app

    _, app, cfg, _, ok_paths, _, work = api
    small = copy.deepcopy(cfg)
    small["serving"]["max_upload_mb"] = 0.001  # ~1 KB; every fixture image is larger
    small["serving"]["prediction_log"] = (work / "small.db").as_posix()
    with TestClient(create_app(small, served=app.state.served)) as c:
        r = _upload(c, ok_paths[0])
        assert r.status_code == 413
        assert len(c.app.state.log) == 0


def test_log_images_opt_in_stores_the_upload(api) -> None:
    from fastapi.testclient import TestClient

    from dr.api import create_app

    _, app, cfg, _, ok_paths, _, work = api
    on = copy.deepcopy(cfg)
    on["serving"]["log_images"] = True
    on["serving"]["image_log_dir"] = (work / "kept").as_posix()
    on["serving"]["prediction_log"] = (work / "kept.db").as_posix()
    with TestClient(create_app(on, served=app.state.served)) as c:
        rid = _upload(c, ok_paths[0]).json()["request_id"]
        stored = Path(c.app.state.log.get(rid)["image_path"])
        assert stored.name == f"{rid}.jpeg" and stored.read_bytes() == ok_paths[0].read_bytes()


def test_health_and_model_info(api) -> None:
    client, app, _, result, _, _, _ = api
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model"] == "dr-referable/1" and h["alias"] == "champion"
    info = client.get("/model-info").json()
    assert info["version"] == "1" and info["alias"] == "champion" and info["model_uri"] == "models:/dr-referable/1"
    assert info["training_run_id"] == result.run_id
    assert info["threshold"] == pytest.approx(app.state.served.threshold)
    assert info["n_embedding"] == 8 and info["image_size"] == 64
