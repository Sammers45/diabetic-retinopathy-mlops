"""MLflow setup shared by every pipeline stage.

Rules this module enforces:

* ``MLFLOW_TRACKING_URI`` (env) always wins over the config, so Colab or CI can point at a remote
  server without editing YAML.
* Whether the *user* set a remote URI is read **before** ``mlflow.set_tracking_uri()`` is called,
  because that call writes ``MLFLOW_TRACKING_URI`` into ``os.environ`` and would otherwise make
  every later check report "user-set".
* With a sqlite backend, artifacts are stored in ``mlartifacts/`` next to the database file, so the
  DB and its artifacts can be copied (e.g. off Google Drive) as one unit.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"
ENV_VAR = "MLFLOW_TRACKING_URI"


@dataclass(frozen=True)
class TrackingInfo:
    """What ``setup_mlflow`` decided, returned so callers can log or print it."""

    tracking_uri: str
    user_set_uri: bool
    experiment_name: str
    experiment_id: str
    artifact_location: str | None


def resolve_tracking_uri(cfg: Mapping[str, Any]) -> tuple[str, bool]:
    """Return ``(uri, user_set)``. Must be called before ``mlflow.set_tracking_uri``."""
    env_uri = os.environ.get(ENV_VAR)
    if env_uri:
        return env_uri, True
    configured = (cfg.get("mlflow") or {}).get("tracking_uri")
    return configured or DEFAULT_TRACKING_URI, False


def sqlite_db_path(uri: str) -> Path | None:
    """Return the database file of a ``sqlite:///`` URI, or ``None`` for other backends.

    ``sqlite:///mlruns.db`` is relative to the working directory; ``sqlite:////abs/path.db``
    (four slashes) and ``sqlite:///C:/path.db`` are absolute.
    """
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return None
    return Path(uri[len(prefix) :]).resolve()


def artifact_root_for(uri: str) -> Path | None:
    """Artifact root next to a sqlite DB (``<db dir>/mlartifacts``), else ``None``."""
    db = sqlite_db_path(uri)
    return None if db is None else db.parent / "mlartifacts"


def setup_mlflow(cfg: Mapping[str, Any], experiment_name: str) -> TrackingInfo:
    """Point MLflow at the right store and select (creating if needed) an experiment."""
    import mlflow

    uri, user_set = resolve_tracking_uri(cfg)  # read BEFORE set_tracking_uri mutates os.environ
    db = sqlite_db_path(uri)
    if db is not None:
        db.parent.mkdir(parents=True, exist_ok=True)
        # Normalise to an absolute path so runs from another working directory hit the same DB.
        uri = "sqlite:///" + db.as_posix()
    mlflow.set_tracking_uri(uri)

    artifact_location: str | None = None
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        root = artifact_root_for(uri)
        if root is not None:
            artifact_location = (root / experiment_name).resolve().as_uri()
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    else:
        experiment_id = experiment.experiment_id
        artifact_location = experiment.artifact_location
    mlflow.set_experiment(experiment_id=experiment_id)
    return TrackingInfo(uri, user_set, experiment_name, experiment_id, artifact_location)


def git_commit(short: bool = False) -> str | None:
    """Current git commit (for run tags), or ``None`` outside a git checkout."""
    cmd = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
