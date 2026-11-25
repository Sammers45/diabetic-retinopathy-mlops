"""Phase 3: patient-level split, local subset and preprocessing (SYNTHETIC data only)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from dr import data as D
from dr import preprocess as P
from dr.config import load_config


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def pipeline(synthetic_data, tmp_path_factory):
    """Manifest -> split -> PNG cache on the synthetic fixtures (no subsetting)."""
    processed = tmp_path_factory.mktemp("p3")
    cfg = load_config(
        "local_cpu",
        overrides=[
            f"paths.raw_dir={synthetic_data.root.as_posix()}",
            f"paths.processed_dir={processed.as_posix()}",
            f"paths.reports_dir={(processed / 'reports').as_posix()}",
            "data.max_images=null",
            "preprocess.image_size=128",
        ],
    )
    df, _ = D.build_manifest(cfg, workers=1)
    df, summary = D.make_splits(df, cfg)
    info = D.cache_preprocessed(df, cfg, workers=1)
    return df, summary, info, cfg


def synthetic_table(n_patients: int, seed: int = 0, prevalence: float = 0.2) -> pd.DataFrame:
    """A manifest-like table without images, for split statistics at realistic scale."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_patients + 1):
        sick = rng.random() < prevalence
        for eye in ("left", "right"):
            level = int(rng.choice([2, 3, 4])) if sick and rng.random() < 0.85 else int(rng.choice([0, 1]))
            rows.append(
                {
                    "image": f"{pid}_{eye}",
                    "patient_id": str(pid),
                    "eye": eye,
                    "level": level,
                    "label": int(level >= 2),
                    "md5": f"{pid}-{eye}",
                    "quality": "ok",
                    "quality_ok": True,
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ acceptance: no leakage
def test_no_patient_or_md5_spans_two_splits(pipeline, synthetic_data) -> None:
    df = pipeline[0]
    used = df[df["split"].isin(D.SPLITS)]
    assert (used.groupby("patient_id")["split"].nunique() == 1).all()
    assert (used.groupby("md5")["split"].nunique() == 1).all()
    a, b = synthetic_data.duplicate
    s = df.set_index("image")["split"]
    g = df.set_index("image")["group_id"]
    assert g[a] == g[b], "patients sharing a byte-identical file are merged into one group"
    assert s[a] == s[b]
    assert pipeline[1]["leakage"] == {
        "patients_in_multiple_splits": 0,
        "md5_in_multiple_splits": 0,
        "groups_in_multiple_splits": 0,
    }
    assert pipeline[1]["merged_md5_groups"] == 1


def test_every_split_has_both_classes(pipeline) -> None:
    for s in D.SPLITS:
        part = pipeline[0][pipeline[0]["split"] == s]
        assert set(part["label"]) == {0, 1}, s


def test_unreadable_image_excluded(pipeline, synthetic_data) -> None:
    s = pipeline[0].set_index("image")["split"]
    assert s[synthetic_data.unreadable] == "excluded"
    assert s[synthetic_data.dark] in D.SPLITS, "poor-but-readable images are kept"


# ------------------------------------------------------------------ split statistics at scale
def test_split_proportions_and_stratification() -> None:
    df = synthetic_table(3000)
    cfg = load_config("local_cpu", overrides=["data.max_images=null"])
    out, summary = D.make_splits(df, cfg)
    n = len(out)
    shares = {s: summary["splits"][s]["images"] / n for s in D.SPLITS}
    assert shares["train"] == pytest.approx(0.70, abs=0.01)
    assert shares["val"] == pytest.approx(0.15, abs=0.01)
    assert shares["test"] == pytest.approx(0.15, abs=0.01)
    prev = [summary["splits"][s]["referable_share"] for s in D.SPLITS]
    assert max(prev) - min(prev) < 0.02, prev
    assert (out.groupby("patient_id")["split"].nunique() == 1).all()


def test_split_is_deterministic_and_seed_dependent() -> None:
    df = synthetic_table(500)
    cfg = load_config("local_cpu", overrides=["data.max_images=null"])
    a = D.make_splits(df, cfg)[0]["split"]
    b = D.make_splits(df, cfg)[0]["split"]
    c = D.make_splits(df, load_config("local_cpu", overrides=["data.max_images=null", "project.seed=7"]))[0]["split"]
    assert (a == b).all()
    assert (a != c).any()


def test_local_subset_is_patient_level_and_stratified() -> None:
    df = synthetic_table(5000)  # 10,000 images
    cfg = load_config("local_cpu", overrides=["data.max_images=3000"])
    out, summary = D.make_splits(df, cfg)
    used = out[out["split"].isin(D.SPLITS)]
    assert len(used) == pytest.approx(3000, rel=0.02)
    assert summary["unused"] == len(out) - len(used)
    # both eyes of a patient are either both used or both unused
    assert (out.groupby("patient_id")["split"].apply(lambda s: s.isin(D.SPLITS).nunique()) == 1).all()
    full_prev = out.groupby("group_id")["label"].max().mean()
    sub_prev = used.groupby("group_id")["label"].max().mean()
    assert sub_prev == pytest.approx(full_prev, abs=0.02)


def test_md5_union_find_merges_chains() -> None:
    df = pd.DataFrame(
        {
            "image": ["1_left", "2_left", "2_right", "3_left", "4_left"],
            "patient_id": ["1", "2", "2", "3", "4"],
            "md5": ["a", "a", "b", "b", "c"],
        }
    )
    groups = D.assign_groups(df)
    assert groups.iloc[0] == groups.iloc[1] == groups.iloc[2] == groups.iloc[3] == "g1"
    assert groups.iloc[4] == "g4"


def test_bad_fractions_rejected() -> None:
    df = synthetic_table(100)
    cfg = load_config("local_cpu", overrides=["data.max_images=null", "data.split.train=0.8"])
    with pytest.raises(D.DataError, match="sum to 1"):
        D.make_splits(df, cfg)


# ------------------------------------------------------------------ preprocessing
PRE = load_config("local_cpu")["preprocess"]


def _fundus(w: int, h: int, brightness: int = 180) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), int(min(w, h) * 0.6), (brightness // 3, brightness // 2, brightness), -1)
    cv2.circle(img, (w // 2 + 40, h // 2), 12, (200, 230, 250), -1)
    return img


@pytest.mark.parametrize("w,h", [(1024, 683), (683, 1024), (500, 500), (320, 240)])
def test_preprocess_shape_dtype_and_rim(w: int, h: int) -> None:
    out = P.preprocess_image(_fundus(w, h), 224, PRE)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
    assert (out[0, 0] == P.PAD_VALUE).all() and (out[-1, -1] == P.PAD_VALUE).all(), "outside the rim is neutral grey"
    centre = out[100:124, 100:124].astype(int)
    assert abs(centre.mean() - 128) < 40, "Ben Graham output is centred near 128"


def test_preprocess_handles_dark_and_empty_images() -> None:
    dark = (_fundus(400, 300) * 0.05).astype(np.uint8)
    assert P.preprocess_image(dark, 64, PRE).shape == (64, 64, 3)
    black = np.zeros((300, 400, 3), np.uint8)
    assert P.preprocess_image(black, 64, PRE).shape == (64, 64, 3)


def test_preprocess_without_ben_graham_keeps_colour() -> None:
    img = _fundus(400, 400)
    out = P.preprocess_image(img, 128, PRE | {"ben_graham": False})
    centre = out[60:68, 60:68].reshape(-1, 3).mean(axis=0)
    assert centre[0] > centre[2], "RGB order: the red channel dominates a fundus"


def test_png_roundtrip_is_bit_exact(tmp_path: Path) -> None:
    out = P.preprocess_image(_fundus(640, 480), 160, PRE)
    P.save_png(out, tmp_path / "x.png")
    assert np.array_equal(P.load_png(tmp_path / "x.png"), out)


def test_bytes_path_equals_file_path(tmp_path: Path) -> None:
    """Serving decodes bytes; caching decodes files. Both must give identical pixels."""
    ok, buf = cv2.imencode(".jpg", _fundus(640, 480), [cv2.IMWRITE_JPEG_QUALITY, 90])
    (tmp_path / "a.jpeg").write_bytes(buf.tobytes())
    from_file = P.preprocess_image(P.read_image(tmp_path / "a.jpeg"), 128, PRE)
    from_bytes = P.preprocess_bytes(buf.tobytes(), 128, PRE)
    assert np.array_equal(from_file, from_bytes)
    assert P.preprocess_bytes(b"garbage", 128, PRE) is None


# ------------------------------------------------------------------ cache
def test_cache_written_for_split_rows_only(pipeline) -> None:
    df, _, info, cfg = pipeline
    used = df[df["split"].isin(D.SPLITS)]
    pngs = sorted(p.stem for p in D.cache_dir(cfg).glob("*.png"))
    assert pngs == sorted(used["image"])
    sample = P.load_png(D.cache_path(cfg, used["image"].iloc[0]))
    assert sample.shape == (128, 128, 3) and sample.dtype == np.uint8
    assert info["signature"]["size"] == 128 and info["failed"] == []


def test_cache_matches_fresh_preprocessing(pipeline, synthetic_data) -> None:
    df, _, _, cfg = pipeline
    row = df[df["split"] == "test"].iloc[0]
    fresh = P.preprocess_bytes((synthetic_data.root / row["path"]).read_bytes(), 128, cfg["preprocess"], cfg["quality"]["mask_threshold"])
    assert np.array_equal(fresh, P.load_png(D.cache_path(cfg, row["image"])))


def test_cache_skips_existing_and_rebuilds_on_setting_change(pipeline) -> None:
    df, _, _, cfg = pipeline
    again = D.cache_preprocessed(df, cfg, workers=1)
    assert again["written"] == 0
    changed = load_config(
        "local_cpu",
        overrides=[
            f"paths.raw_dir={cfg['paths']['raw_dir']}",
            f"paths.processed_dir={cfg['paths']['processed_dir']}",
            "preprocess.image_size=128",
            "preprocess.rim_fraction=0.9",
        ],
    )
    rebuilt = D.cache_preprocessed(df, changed, workers=1)
    assert rebuilt["written"] == rebuilt["images"]
    D.cache_preprocessed(df, cfg, workers=1)  # restore for other tests


def test_visual_check_written(pipeline, tmp_path: Path) -> None:
    df, _, _, cfg = pipeline
    out = D.save_visual_check(df, cfg, tmp_path / "check.png", n=4)
    img = P.load_png(out)
    assert img.shape[1] == 128 * 2 + 8 and img.shape[0] >= 128 * 2


def test_text_files_are_clean_utf8(repo_root: Path) -> None:
    """Guard against double-encoded UTF-8, e.g. a '≥' read as ANSI and re-saved (a Windows
    PowerShell 5.1 round trip turned README.md into garbage once)."""
    # Built with chr() so this file does not itself contain the patterns it searches for.
    codes = [(0xE2, 0x20AC), (0xE2, 0x2030), (0xE2, 0x161), (0xC3,), (0xFFFD,)]
    mojibake = tuple("".join(map(chr, c)) for c in codes)
    skip = {".venv", "data", ".git", "mlartifacts", ".pytest_cache"}
    bad = []
    for path in repo_root.rglob("*"):
        if path.suffix not in {".md", ".py", ".yaml", ".yml", ".ipynb", ".txt", ".ps1"} or skip & set(path.relative_to(repo_root).parts):
            continue
        text = path.read_text(encoding="utf-8")
        if any(m in text for m in mojibake):
            bad.append(str(path.relative_to(repo_root)))
    assert bad == [], f"double-encoded characters in {bad}"


def test_cli_end_to_end(synthetic_data, tmp_path: Path, capsys) -> None:
    code = D.main(
        [
            "--workers", "1",
            "--set", f"paths.raw_dir={synthetic_data.root.as_posix()}",
            "--set", f"paths.processed_dir={(tmp_path / 'p').as_posix()}",
            "--set", f"paths.reports_dir={(tmp_path / 'r').as_posix()}",
            "--set", "preprocess.image_size=64",
        ]
    )
    assert code == 0, capsys.readouterr().err
    summary = json.loads((tmp_path / "p" / D.SPLIT_SUMMARY).read_text(encoding="utf-8"))
    assert summary["leakage"]["patients_in_multiple_splits"] == 0
    manifest = pd.read_csv(tmp_path / "p" / D.MANIFEST)
    assert {"group_id", "group_label", "split"} <= set(manifest.columns)
    assert (tmp_path / "r" / "local" / "preprocess_check.png").exists()
