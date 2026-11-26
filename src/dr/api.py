"""FastAPI service for the registered ``dr-referable`` model.

Endpoints:

* ``POST /predict``: multipart image upload. Returns ``p_referable``, the yes/no decision,
  threshold, quality flag, model version and a request ID. Every scored upload is appended to the
  prediction log.
* ``POST /feedback``: ``{"request_id", "grade"}`` attaches a clinician's ICDR grade (0-4) to a
  logged prediction, so the monitor can measure real performance later.
* ``GET /health`` and ``GET /model-info``.

Design choices:

* At startup, ``models:/<name>@<alias>`` is resolved to a concrete version, which is then loaded
  by version. If the alias moves while the service runs, responses still report the version that
  actually scored them. A reload (restart) picks up the new alias target.
* Scoring goes through the registered pyfunc, the same object the tests compare against
  training-time predictions. The API has no preprocessing of its own, so serving cannot drift
  from training.
* Uploads are checked (content type, size, empty) before any decoding. An upload that passes
  those checks but cannot be decoded is logged as ``poor:unreadable`` and answered with 422, so
  the monitor still sees the failure rate.

Not a medical device: every response carries that disclaimer.
"""

from __future__ import annotations

import argparse
import base64
import sys
import threading
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from dr.config import ConfigError, add_config_args, config_from_args, get, resolve_path
from dr.prediction_log import PredictionLog, utc_now
from dr.preprocess import STAT_COLUMNS, md5_bytes
from dr.serving_model import CLASS_NAMES, UNREADABLE

DISCLAIMER = "Research prototype. Not a medical device; not for diagnosis or screening."
EXTENSIONS = {"image/jpeg": ".jpeg", "image/png": ".png", "image/tiff": ".tiff", "image/bmp": ".bmp"}


class ModelNotAvailable(RuntimeError):
    """No model is registered under the configured name and alias."""


@dataclass
class ServedModel:
    """The loaded pyfunc plus the registry facts reported by ``/model-info``."""

    pyfunc: Any
    name: str
    alias: str
    version: str
    model_uri: str
    tracking_uri: str
    meta: dict[str, Any]
    loaded_at: str = field(default_factory=utc_now)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alias": self.alias,
            "version": self.version,
            "model_uri": self.model_uri,
            "threshold": self.threshold,
            "training_run_id": self.meta.get("training_run_id"),
            "model_family": self.meta.get("model_family"),
            "image_size": self.meta.get("image_size"),
            "n_embedding": self.meta.get("n_emb"),
            "class_names": self.meta.get("class_names", CLASS_NAMES),
            "tracking_uri": self.tracking_uri,
            "loaded_at": self.loaded_at,
        }

    def predict_bytes(self, data: bytes) -> dict[str, Any]:
        """Score one encoded image through the registered pyfunc (one row in, one row out)."""
        frame = pd.DataFrame({"image_b64": [base64.b64encode(data).decode("ascii")]})
        with self.lock:  # one inference at a time: predictable latency and memory on a small box
            out = self.pyfunc.predict(frame)
        return out.iloc[0].to_dict()


def load_served_model(cfg: Mapping[str, Any]) -> ServedModel:
    """Resolve ``<registered_model>@<alias>`` to a version and load that version."""
    import mlflow

    from dr.tracking import configure_tracking

    uri, _ = configure_tracking(cfg)
    name = str(get(cfg, "mlflow.registered_model"))
    alias = str(get(cfg, "serving.model_alias", "champion"))
    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except Exception as exc:  # MlflowException: model or alias missing
        raise ModelNotAvailable(
            f"No model registered as {name}@{alias} in {uri}. Train and register one first (make train), "
            f"or point MLFLOW_TRACKING_URI at the tracking server that holds it."
        ) from exc
    model_uri = f"models:/{name}/{mv.version}"
    pyfunc = mlflow.pyfunc.load_model(model_uri)
    meta = dict(pyfunc.unwrap_python_model().meta)
    return ServedModel(pyfunc=pyfunc, name=name, alias=alias, version=str(mv.version), model_uri=model_uri, tracking_uri=uri, meta=meta)


# --------------------------------------------------------------------------- schemas
class ModelRef(BaseModel):
    name: str
    version: str
    alias: str


class PredictResponse(BaseModel):
    request_id: str
    timestamp: str
    p_referable: float
    prediction: int = Field(description="1 = referable (p_referable >= threshold)")
    label: str
    threshold: float
    quality: str
    model: ModelRef
    disclaimer: str = DISCLAIMER


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    grade: int = Field(ge=0, le=4, description="ICDR grade 0-4 from a clinician")


class FeedbackResponse(BaseModel):
    request_id: str
    grade: int
    label: int
    prediction: int | None
    feedback_at: str


# --------------------------------------------------------------------------- app
def create_app(cfg: Mapping[str, Any], served: ServedModel | None = None) -> FastAPI:
    """Build the app. ``served`` is injectable for tests; otherwise it loads at startup."""
    scfg = dict(get(cfg, "serving", {}) or {})
    max_bytes = int(float(scfg.get("max_upload_mb", 10)) * 1024 * 1024)
    allowed = {str(t).lower() for t in scfg.get("allowed_content_types", list(EXTENSIONS))}
    log_images = bool(scfg.get("log_images", False))
    image_dir = resolve_path(cfg, "serving.image_log_dir") if log_images else None
    min_grade = int(get(cfg, "data.referable_min_grade", 2))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.served = served if served is not None else load_served_model(cfg)
        app.state.log = PredictionLog(resolve_path(cfg, "serving.prediction_log"))
        yield

    app = FastAPI(
        title="Referable DR classifier",
        description=f"Referable diabetic retinopathy (ICDR >= 2) from one fundus photograph. {DISCLAIMER}",
        version="0.6.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        m: ServedModel = app.state.served
        return {"status": "ok", "model": f"{m.name}/{m.version}", "alias": m.alias, "logged_predictions": len(app.state.log)}

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        return app.state.served.info()

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        file: UploadFile = File(..., description="Fundus photograph (JPEG, PNG, TIFF or BMP)"),
        patient_id: str | None = Form(None, max_length=64),
        eye: str | None = Form(None, pattern="^(left|right)$"),
    ) -> Any:
        content_type = (file.content_type or "").split(";")[0].strip().lower()
        if content_type not in allowed:
            raise HTTPException(415, f"Unsupported content type {content_type or 'none'!r}; allowed: {sorted(allowed)}")
        data = file.file.read(max_bytes + 1)  # never read more than the limit
        if len(data) > max_bytes:
            raise HTTPException(413, f"Upload larger than {max_bytes // (1024 * 1024)} MB")
        if not data:
            raise HTTPException(400, "Empty upload")

        m: ServedModel = app.state.served
        request_id = uuid.uuid4().hex
        timestamp = utc_now()
        row = m.predict_bytes(data)
        image_path = None
        if image_dir is not None:
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{request_id}{EXTENSIONS.get(content_type, '.img')}"
            image_path.write_bytes(data)
        record = {
            "request_id": request_id,
            "timestamp": timestamp,
            "source": "api",
            "model_name": m.name,
            "model_version": m.version,
            "md5": md5_bytes(data),
            "patient_id": patient_id,
            "eye": eye,
            "filename": (file.filename or "")[:255] or None,
            "image_path": str(image_path) if image_path else None,
            **{k: v for k, v in row.items() if k in ("p_referable", "prediction", "threshold", "quality", "error", *STAT_COLUMNS) or k.startswith("emb_")},
        }
        app.state.log.append([record])

        if row.get("quality") == UNREADABLE or pd.isna(row.get("p_referable")):
            return JSONResponse(
                status_code=422,
                content={"detail": f"Not a readable image: {row.get('error') or 'decode failed'}", "request_id": request_id, "quality": UNREADABLE},
            )
        prediction = int(row["prediction"])
        return PredictResponse(
            request_id=request_id,
            timestamp=timestamp,
            p_referable=float(row["p_referable"]),
            prediction=prediction,
            label=CLASS_NAMES[prediction],
            threshold=float(row["threshold"]),
            quality=str(row["quality"]),
            model=ModelRef(name=m.name, version=m.version, alias=m.alias),
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(body: FeedbackRequest) -> Any:
        updated = app.state.log.add_feedback(body.request_id, body.grade, min_grade)
        if updated is None:
            raise HTTPException(404, f"Unknown request_id {body.request_id!r}")
        return FeedbackResponse(
            request_id=updated["request_id"],
            grade=updated["grade"],
            label=updated["label"],
            prediction=updated["prediction"],
            feedback_at=updated["feedback_at"],
        )

    return app


def main(argv: list[str] | None = None) -> int:
    """``python -m dr.api``: serve the registered model with uvicorn."""
    import uvicorn

    parser = add_config_args(argparse.ArgumentParser(description="Serve dr-referable over HTTP."))
    parser.add_argument("--host", default=None, help="Bind address (default: serving.host)")
    parser.add_argument("--port", type=int, default=None, help="Port (default: serving.port)")
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
        served = load_served_model(cfg)  # fail fast with a clear message, before uvicorn starts
    except (ConfigError, ModelNotAvailable) as exc:
        print(f"Cannot start the API: {exc}", file=sys.stderr)
        return 2
    m = served.info()
    print(f"Serving {m['name']} v{m['version']} (@{m['alias']}), threshold {m['threshold']:.4f}, from {m['tracking_uri']}")
    host = args.host or str(get(cfg, "serving.host", "127.0.0.1"))
    port = args.port or int(get(cfg, "serving.port", 8000))
    uvicorn.run(create_app(cfg, served=served), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
