"""Phase 2: ingest, validation, image statistics and quality flags (SYNTHETIC data only)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from dr import data as D
from dr import eda
from dr import preprocess as P
from dr.config import load_config


@pytest.fixture(scope="module")
def built(synthetic_data, tmp_path_factory):
    """Run the Phase 2 pipeline once on the synthetic fixtures."""
    processed = tmp_path_factory.mktemp("processed")
    cfg = load_config(
        "local_cpu",
        overrides=[f"paths.raw_dir={synthetic_data.root.as_posix()}", f"paths.processed_dir={processed.as_posix()}"],
    )
    df, report = D.build_manifest(cfg, workers=1)
    D.write_outputs(df, report, processed)
    return df, report, processed, cfg


# ------------------------------------------------------------------ acceptance criteria
def test_unreadable_and_dark_images_flagged(built, synthetic_data) -> None:
    df = built[0].set_index("image")
    assert df.loc[synthetic_data.unreadable, "quality"] == "poor:unreadable"
    assert not df.loc[synthetic_data.unreadable, "readable"]
    assert df.loc[synthetic_data.dark, "quality"] == "poor:too_dark"
    normal = df.drop(index=[synthetic_data.unreadable, synthetic_data.dark])
    assert (normal["quality"] == "ok").all(), normal.loc[normal["quality"] != "ok", "quality"]


def test_labels_match_level_ge_2(built, synthetic_data) -> None:
    df = built[0]
    assert (df["label"] == (df["level"] >= 2).astype(int)).all()
    truth = pd.read_csv(synthetic_data.labels_csv).set_index("image")["level"]
    assert (df.set_index("image")["level"] == truth.loc[df["image"]]).all()


def test_manifest_and_report_written(built) -> None:
    df, report, processed, _ = built
    on_disk = pd.read_csv(processed / D.MANIFEST, dtype={"patient_id": str})
    assert list(on_disk.columns) == D.MANIFEST_COLUMNS
    assert len(on_disk) == 60 and on_disk["patient_id"].nunique() == 30
    assert set(on_disk["eye"]) == {"left", "right"}
    assert not on_disk["path"].str.contains(":").any(), "paths are stored relative to raw_dir"
    saved = json.loads((processed / D.QUALITY_REPORT).read_text(encoding="utf-8"))
    assert saved["n_images"] == 60
    assert saved["quality_counts"]["poor:unreadable"] == 1
    assert saved["md5_duplicates"]["md5_groups_spanning_patients"] == 1
    assert "thresholds" in saved and saved["labels_csv"] == "trainLabels.csv"


def test_duplicate_md5_detected(built, synthetic_data) -> None:
    df = built[0].set_index("image")
    a, b = synthetic_data.duplicate
    assert df.loc[a, "md5"] == df.loc[b, "md5"]
    assert df.loc[a, "patient_id"] != df.loc[b, "patient_id"]


# ------------------------------------------------------------------ discovery + validation
def test_parse_name() -> None:
    assert D.parse_name("10_left") == ("10", "left")
    assert D.parse_name("44349_right") == ("44349", "right")
    for bad in ("10-left", "abc_left", "10_up", "10_left_x", ""):
        assert D.parse_name(bad) is None


def test_find_labels_csv_prefers_most_rows(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    pd.DataFrame({"image": ["1_left", "1_right"], "level": [0, 2]}).to_csv(tmp_path / "a" / "trainLabels.csv", index=False)
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(tmp_path / "other.csv", index=False)
    pd.DataFrame({"image": ["1_left"], "level": [0]}).to_csv(tmp_path / "trainLabels_cropped.csv", index=False)
    assert D.find_labels_csv(tmp_path).name == "trainLabels.csv"
    assert D.find_labels_csv(tmp_path, "trainLabels_cropped.csv").name == "trainLabels_cropped.csv"
    with pytest.raises(D.DataError):
        D.find_labels_csv(tmp_path, "nope.csv")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(D.DataError, match="dr.download"):
        D.find_labels_csv(empty)


def test_validation_reports_every_issue(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        {
            "image": ["1_left", "1_right", "1_right", "2_left", "2_left", "3_left", "bad-name", "4_left", "5_left"],
            "level": [0, 2, 2, 1, 3, 7, 0, 1, 4],
        }
    )
    csv = tmp_path / "trainLabels.csv"
    labels.to_csv(csv, index=False)
    index = {s: tmp_path / f"{s}.jpeg" for s in ["1_left", "1_right", "2_left", "3_left", "bad-name", "5_left", "9_left"]}
    df, issues = D.load_and_validate(csv, index, referable_min_grade=2)
    assert issues["duplicate_label_rows"] == 2
    assert issues["duplicate_conflicting_grades"] == ["2_left"]
    assert issues["invalid_grades"] == ["3_left"]
    assert issues["unparseable_names"] == ["bad-name"]
    assert issues["missing_images"] == ["4_left"]
    assert issues["images_without_label"] == 1  # 9_left
    assert sorted(df["image"]) == ["1_left", "1_right", "5_left"]
    assert df.set_index("image")["label"].to_dict() == {"1_left": 0, "1_right": 1, "5_left": 1}


def test_index_prefers_best_directory_and_excludes(tmp_path: Path) -> None:
    for d, stems in {"orig": ["1_left", "1_right", "2_left"], "partial": ["1_left"], "x_cropped": ["1_left", "1_right", "2_left"]}.items():
        (tmp_path / d).mkdir()
        for s in stems:
            (tmp_path / d / f"{s}.jpeg").write_bytes(b"x")
    wanted = {"1_left", "1_right", "2_left"}
    index, info = D.index_images(tmp_path, wanted, exclude_dirs=["cropped"])
    assert all(p.parent.name == "orig" for p in index.values())
    assert info["stem_conflicts_resolved"] == 1
    assert "x_cropped" not in info["image_dirs"]


# ------------------------------------------------------------------ stats + quality
QCFG = load_config("local_cpu")["quality"]


def _disc(brightness: int, size: int = 400, blur: int = 0) -> np.ndarray:
    img = np.zeros((size, size, 3), np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.45), (brightness // 3, brightness // 2, brightness), -1)
    rng = np.random.default_rng(0)
    for _ in range(40):  # texture so the image is not trivially "blurred"
        p = tuple(int(v) for v in rng.integers(size // 4, 3 * size // 4, 2))
        cv2.circle(img, p, 3, (0, 0, max(0, brightness - 80)), -1)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def test_quality_flags_on_constructed_images() -> None:
    def flag(img):
        return P.quality_flag(P.image_stats(img, QCFG), QCFG)

    assert flag(_disc(200)) == "ok"
    assert flag(np.full((300, 300, 3), 255, np.uint8)) == "poor:overexposed"
    assert flag(_disc(200, blur=12)) == "poor:blurred"
    assert flag((_disc(200) * 0.06).astype(np.uint8)) == "poor:too_dark"
    tiny = np.zeros((400, 400, 3), np.uint8)
    cv2.circle(tiny, (200, 200), 30, (120, 160, 230), -1)
    assert flag(tiny) == "poor:no_fundus"
    assert P.quality_flag(P.image_stats(None, QCFG), QCFG, readable=False) == "poor:unreadable"


def test_decode_rejects_garbage() -> None:
    assert P.decode_image(b"") is None
    assert P.decode_image(b"not an image at all") is None
    ok, buf = cv2.imencode(".png", _disc(150))
    assert P.decode_image(buf.tobytes()).shape == (400, 400, 3)


def test_stats_are_resolution_independent() -> None:
    small = _disc(180, size=400)
    big = cv2.resize(small, (1600, 1600), interpolation=cv2.INTER_CUBIC)
    s1, s2 = P.image_stats(small, QCFG), P.image_stats(big, QCFG)
    assert abs(s1["brightness"] - s2["brightness"]) < 2
    assert abs(s1["fundus_fraction"] - s2["fundus_fraction"]) < 0.02


def test_stats_cache_reused_and_thresholds_reapplied(synthetic_data, tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    shutil.copytree(synthetic_data.root, raw)
    cfg = load_config("local_cpu", overrides=[f"paths.raw_dir={raw.as_posix()}", f"paths.processed_dir={(tmp_path / 'p').as_posix()}"])
    calls = {"n": 0}
    real = D._analyse_file

    def counting(args):
        calls["n"] += 1
        return real(args)

    monkeypatch.setattr(D, "_analyse_file", counting)
    D.build_manifest(cfg, workers=1)
    assert calls["n"] == 60
    df2, _ = D.build_manifest(cfg, workers=1)
    assert calls["n"] == 60, "second run must come entirely from the cache"
    strict = load_config("local_cpu", overrides=[*[f"paths.{k}={v}" for k, v in [("raw_dir", raw.as_posix()), ("processed_dir", (tmp_path / 'p').as_posix())]], "quality.min_sharpness=1e9"])
    df3, _ = D.build_manifest(strict, workers=1)
    assert calls["n"] == 60, "changing a threshold must not rescan"
    assert (df3.loc[df3["readable"], "quality"] != "ok").all()


def test_cli_missing_raw_dir(tmp_path: Path, capsys) -> None:
    code = D.main(["--set", f"paths.raw_dir={(tmp_path / 'nothing').as_posix()}"])
    assert code == 2 and "dr.download" in capsys.readouterr().err


# ------------------------------------------------------------------ EDA helpers
def test_eda_plots_run_on_synthetic_manifest(built, synthetic_data, tmp_path: Path) -> None:
    df = eda.load_manifest(built[2] / D.MANIFEST)
    written = eda.save_aggregate_plots(df, tmp_path / "eda")
    assert {p.name for p in written} == {f"{k}.png" for k in eda.AGGREGATE_PLOTS}
    assert all(p.stat().st_size > 1000 for p in written)
    fig = eda.show_examples(df, synthetic_data.root, by="quality", n=2)
    assert len(fig.axes) == 2 * df["quality"].nunique()


def test_eda_notebook_in_sync_and_output_free() -> None:
    """01_eda.ipynb is generated, carries no outputs (no real images in git) and only calls src/."""
    import json as _json

    from build_eda_notebook import OUT, build

    on_disk = _json.loads(OUT.read_text(encoding="utf-8"))
    assert on_disk == build(), "re-run scripts/build_eda_notebook.py"
    for cell in on_disk["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            src = "".join(cell["source"])
            assert "def " not in src and "class " not in src, "core logic belongs in src/dr"
