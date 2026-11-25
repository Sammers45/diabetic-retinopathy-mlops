"""Phase 0: config system, overrides, MLflow helper, synthetic fixtures (all offline)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import pandas as pd
import pytest
import yaml

from dr import config as C
from dr import tracking


# ---------------------------------------------------------------- config & profiles
@pytest.mark.parametrize("profile", C.PROFILES)
def test_both_profiles_load(profile: str) -> None:
    cfg = C.load_config(profile=profile)
    assert cfg["meta"]["profile"] == profile
    assert "profile" not in cfg, "profile name must live under meta.profile, not a top-level key"
    for section in ("data", "preprocess", "model", "train", "evaluate", "reference", "mlflow"):
        assert isinstance(cfg[section], dict)


def test_profiles_differ_where_expected() -> None:
    local = C.load_config("local_cpu")
    colab = C.load_config("colab")
    assert local["preprocess"]["image_size"] == 224
    assert colab["preprocess"]["image_size"] == 384
    assert local["data"]["max_images"] == 3000
    assert colab["data"]["max_images"] is None
    assert colab["train"]["mixed_precision"] is True
    assert local["mlflow"]["tracking_uri"] == "sqlite:///mlruns.db"


def test_numeric_types_survive_yaml() -> None:
    """PyYAML reads `1e-3` as a string; learning rates must be floats."""
    cfg = C.load_config("local_cpu")
    assert isinstance(cfg["train"]["head_lr"], float)
    assert isinstance(cfg["train"]["finetune_lr"], float)
    assert isinstance(cfg["data"]["split"]["train"], float)
    assert abs(sum(cfg["data"]["split"].values()) - 1.0) < 1e-9


def test_profile_resolution_order(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DR_PROFILE", "COLAB_RELEASE_TAG", "COLAB_GPU"):
        monkeypatch.delenv(var, raising=False)
    assert C.resolve_profile() == "local_cpu"
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab_2026")
    assert C.resolve_profile() == "colab"
    monkeypatch.setenv("DR_PROFILE", "local_cpu")
    assert C.resolve_profile() == "local_cpu"  # env beats auto-detection
    assert C.resolve_profile("colab") == "colab"  # explicit beats env
    with pytest.raises(C.ConfigError):
        C.resolve_profile("gpu_cluster")


def test_overrides_are_typed_and_applied() -> None:
    cfg = C.load_config(
        "local_cpu",
        overrides=[
            "train.batch_size=8",
            "train.head_lr=0.01",
            "model.pretrained=false",
            "data.kaggle_dataset=owner/some-dataset",
            "data.max_images=null",
            "data.split.val=0.2",
        ],
    )
    assert cfg["train"]["batch_size"] == 8
    assert cfg["train"]["head_lr"] == 0.01
    assert cfg["model"]["pretrained"] is False
    assert cfg["data"]["kaggle_dataset"] == "owner/some-dataset"
    assert cfg["data"]["max_images"] is None
    assert cfg["data"]["split"]["val"] == 0.2


@pytest.mark.parametrize(
    "bad",
    ["train.batchsize=8", "nosuch.key=1", "train=1", "train.batch_size", "train..batch_size=1"],
)
def test_bad_overrides_raise(bad: str) -> None:
    with pytest.raises(C.ConfigError):
        C.load_config("local_cpu", overrides=[bad])


def test_cli_parser_accepts_profile_and_set() -> None:
    import argparse

    parser = C.add_config_args(argparse.ArgumentParser())
    args = parser.parse_args(["--profile", "colab", "--set", "train.batch_size=4", "--set", "model.family=efficientnet_b3"])
    cfg = C.config_from_args(args)
    assert cfg["meta"]["profile"] == "colab"
    assert cfg["train"]["batch_size"] == 4
    assert cfg["model"]["family"] == "efficientnet_b3"


def test_config_cli_prints_yaml(capsys: pytest.CaptureFixture[str]) -> None:
    assert C.main(["--profile", "local_cpu", "--set", "train.batch_size=2"]) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["train"]["batch_size"] == 2
    assert C.main(["--set", "bogus.key=1"]) == 2


def test_yaml_boolean_keys_rejected(tmp_path: Path) -> None:
    """An unquoted `yes:` key becomes True in YAML 1.1; the loader must refuse it."""
    (tmp_path / "profiles").mkdir()
    (tmp_path / "base.yaml").write_text("answers:\n  yes: 1\n  no: 0\n", encoding="utf-8")
    (tmp_path / "profiles" / "local_cpu.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(C.ConfigError, match="not a string"):
        C.load_config("local_cpu", config_dir=tmp_path)


def test_all_shipped_yaml_has_string_keys(repo_root: Path) -> None:
    for path in (repo_root / "configs").rglob("*.yaml"):
        C._read_yaml(path)  # raises on non-string keys


def test_flatten_and_get() -> None:
    cfg = C.load_config("local_cpu")
    flat = C.flatten(cfg)
    assert flat["preprocess.image_size"] == 224
    assert flat["meta.profile"] == "local_cpu"
    assert C.get(cfg, "mlflow.registered_model") == "dr-referable"
    assert C.get(cfg, "missing.key", "dflt") == "dflt"


# ---------------------------------------------------------------- MLflow helper
def test_tracking_uri_precedence(clean_mlflow_env: None) -> None:
    cfg = C.load_config("local_cpu")
    assert tracking.resolve_tracking_uri(cfg) == ("sqlite:///mlruns.db", False)
    assert tracking.resolve_tracking_uri({"mlflow": {"tracking_uri": None}}) == (
        tracking.DEFAULT_TRACKING_URI,
        False,
    )
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.example:5000"
    assert tracking.resolve_tracking_uri(cfg) == ("http://mlflow.example:5000", True)


def test_sqlite_paths() -> None:
    assert tracking.sqlite_db_path("http://host:5000") is None
    db = tracking.sqlite_db_path("sqlite:///some/dir/mlruns.db")
    assert db is not None and db.name == "mlruns.db" and db.is_absolute()
    assert tracking.artifact_root_for("sqlite:///some/dir/mlruns.db") == db.parent / "mlartifacts"


def test_setup_mlflow_sqlite_artifacts_next_to_db(tmp_path: Path, clean_mlflow_env: None) -> None:
    import mlflow

    db = tmp_path / "store" / "mlruns.db"
    cfg = C.load_config("local_cpu", overrides=[f"mlflow.tracking_uri=sqlite:///{db.as_posix()}"])
    info = tracking.setup_mlflow(cfg, "dr-test")

    # The flag was read before set_tracking_uri, so it must not report a user-set URI even if
    # MLflow has since written MLFLOW_TRACKING_URI into os.environ.
    assert info.user_set_uri is False
    assert db.exists()
    assert info.artifact_location is not None
    expected_root = (db.parent / "mlartifacts").resolve().as_uri()
    assert info.artifact_location.startswith(expected_root)

    with mlflow.start_run() as run:
        mlflow.log_param("p", 1)
        mlflow.log_metric("m", 0.5)
    stored = mlflow.get_run(run.info.run_id)
    assert stored.data.params["p"] == "1"
    assert stored.info.artifact_uri.startswith(expected_root)

    # Idempotent: a second call reuses the experiment.
    again = tracking.setup_mlflow(cfg, "dr-test")
    assert again.experiment_id == info.experiment_id


# ---------------------------------------------------------------- synthetic fixtures
def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_fixture_layout_and_labels(synthetic_data) -> None:
    fs = synthetic_data
    labels = pd.read_csv(fs.labels_csv)
    assert list(labels.columns) == ["image", "level"]
    assert len(labels) == fs.n_images == 60
    assert labels["level"].between(0, 4).all()
    assert (labels["level"] >= 2).any() and (labels["level"] < 2).any(), "both classes present"
    for stem in labels["image"]:
        pid, eye = stem.split("_")
        assert pid.isdigit() and eye in {"left", "right"}
        assert (fs.image_dir / f"{stem}.jpeg").exists()
    assert "SYNTHETIC" in (fs.root / "SYNTHETIC_DATA_README.txt").read_text(encoding="utf-8")


def test_fixture_edge_cases(synthetic_data) -> None:
    fs = synthetic_data
    a, b = fs.duplicate
    assert a.split("_")[0] != b.split("_")[0], "duplicate must span two patients"
    assert _md5(fs.image_dir / f"{a}.jpeg") == _md5(fs.image_dir / f"{b}.jpeg")

    assert cv2.imread(str(fs.image_dir / f"{fs.unreadable}.jpeg")) is None

    dark = cv2.imread(str(fs.image_dir / f"{fs.dark}.jpeg"))
    assert dark is not None and dark.mean() < 10

    normal = cv2.imread(str(fs.image_dir / f"{a}.jpeg"))
    assert normal is not None and normal.mean() > 20


def test_fixtures_deterministic(tmp_path: Path) -> None:
    from make_synthetic_fixtures import generate_fixtures

    one = generate_fixtures(tmp_path / "a", n_patients=8, seed=3)
    two = generate_fixtures(tmp_path / "b", n_patients=8, seed=3)
    assert one.grades == two.grades
    for stem in one.grades:
        assert _md5(one.image_dir / f"{stem}.jpeg") == _md5(two.image_dir / f"{stem}.jpeg")


# ---------------------------------------------------------------- Colab notebook
def test_colab_notebook_in_sync_with_generator(repo_root: Path) -> None:
    """The committed notebook must match scripts/build_colab_notebook.py (re-run it if this fails)."""
    import json

    from build_colab_notebook import OUT, build

    nb = build()
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    assert json.loads(OUT.read_text(encoding="utf-8")) == nb
    sources = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "KAGGLE_API_TOKEN" in sources and "DR_PROFILE" in sources
