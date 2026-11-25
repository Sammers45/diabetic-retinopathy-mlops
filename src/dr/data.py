"""Ingest, validate and profile the raw data (``python -m dr.data``).

Phase 2 steps (this module):

1. Find the labels CSV (columns ``image``, ``level``) under ``paths.raw_dir``.
2. Index images recursively by file stem and parse ``patient_id`` / ``eye`` from the name.
3. Validate: missing images, duplicate label rows, invalid grades, unparseable names.
4. Compute per-image statistics, MD5 and a quality flag in parallel (cached between runs).
5. Write ``data/processed/manifest.csv`` and ``data/processed/data_quality.json``.

Phase 3 adds the patient-level split and the preprocessed PNG cache to the same CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dr import preprocess as P
from dr.config import ConfigError, add_config_args, config_from_args, get, resolve_path

IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
NAME_RE = re.compile(r"^(?P<patient>\d+)_(?P<eye>left|right)$")
VALID_GRADES = {0, 1, 2, 3, 4}
STATS_CACHE = "image_stats_cache.parquet"
MANIFEST = "manifest.csv"
QUALITY_REPORT = "data_quality.json"

# Ordered manifest columns (documented in data/README.md).
MANIFEST_COLUMNS = [
    "image",
    "patient_id",
    "eye",
    "level",
    "label",
    "path",
    "file_bytes",
    "md5",
    "readable",
    *P.STAT_COLUMNS,
    "aspect_ratio",
    "dark_p99",
    "quality",
    "quality_ok",
]


class DataError(RuntimeError):
    """Data problems the user has to fix (no labels CSV, no images...)."""


# --------------------------------------------------------------------------- discovery
def find_labels_csv(raw_dir: Path, explicit: str | None = None) -> Path:
    """Locate the labels CSV: explicit setting, else the CSV with ``image``+``level`` and most rows.

    Most rows wins because derived CSVs (e.g. a "cropped" subset) drop a few images; the name
    ``trainLabels.csv`` breaks ties.
    """
    if explicit:
        path = (raw_dir / explicit) if not Path(explicit).is_absolute() else Path(explicit)
        if not path.is_file():
            raise DataError(f"data.labels_csv={explicit!r} not found at {path}.")
        return path
    candidates: list[tuple[int, int, Path]] = []
    for csv in raw_dir.rglob("*.csv"):
        try:
            head = pd.read_csv(csv, nrows=5)
        except Exception:
            continue
        if {"image", "level"} <= set(head.columns):
            with csv.open("rb") as fh:
                rows = sum(1 for _ in fh) - 1
            candidates.append((rows, int(csv.name.lower() == "trainlabels.csv"), csv))
    if not candidates:
        raise DataError(
            f"No labels CSV with columns 'image' and 'level' found under {raw_dir}. "
            "Run `python -m dr.download` first or set --set data.labels_csv=<path>."
        )
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return candidates[0][2]


def index_images(raw_dir: Path, wanted: set[str], exclude_dirs: Iterable[str] = ()) -> tuple[dict[str, Path], dict[str, Any]]:
    """Map file stem -> path, searching recursively.

    When the same stem exists in several directories (e.g. original and pre-cropped copies),
    directories are ranked by how many labelled stems they cover and the best one wins. Excluded
    directory substrings are skipped entirely.
    """
    excl = [e.lower() for e in exclude_dirs]
    by_dir: dict[Path, dict[str, Path]] = {}
    for root, dirs, files in os.walk(raw_dir):
        dirs[:] = [d for d in dirs if not any(e in d.lower() for e in excl)]
        for name in files:
            p = Path(root, name)
            if p.suffix.lower() in IMAGE_EXTS:
                by_dir.setdefault(Path(root), {})[p.stem] = p
    ranked = sorted(by_dir.items(), key=lambda kv: len(wanted & kv[1].keys()), reverse=True)
    index: dict[str, Path] = {}
    conflicts = 0
    for _, files in ranked:
        for stem, path in files.items():
            if stem in index:
                conflicts += 1
            else:
                index[stem] = path
    info = {
        "image_dirs": {str(d.relative_to(raw_dir)): len(f) for d, f in ranked},
        "stem_conflicts_resolved": conflicts,
        "excluded_dir_patterns": list(excl),
    }
    return index, info


def parse_name(stem: str) -> tuple[str, str] | None:
    """``"10_left"`` -> ``("10", "left")``; ``None`` if the name does not follow the convention."""
    m = NAME_RE.match(stem.strip())
    return (m["patient"], m["eye"]) if m else None


# --------------------------------------------------------------------------- validation
def load_and_validate(labels_csv: Path, index: Mapping[str, Path], referable_min_grade: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the labels table and report every problem (problems are dropped, never silently)."""
    raw = pd.read_csv(labels_csv, dtype={"image": str})
    raw["image"] = raw["image"].str.strip()
    issues: dict[str, Any] = {"label_rows": len(raw)}

    dup_mask = raw.duplicated("image", keep=False)
    dups = raw[dup_mask]
    conflicting = dups.groupby("image")["level"].nunique()
    issues["duplicate_label_rows"] = int(raw.duplicated("image").sum())
    issues["duplicate_conflicting_grades"] = sorted(conflicting[conflicting > 1].index.tolist())[:50]
    df = raw[~raw["image"].isin(issues["duplicate_conflicting_grades"])].drop_duplicates("image").copy()

    level = pd.to_numeric(df["level"], errors="coerce")
    bad_grade = ~level.isin(list(VALID_GRADES))
    issues["invalid_grades"] = df.loc[bad_grade, "image"].tolist()[:50]
    issues["invalid_grades_count"] = int(bad_grade.sum())
    df = df[~bad_grade].copy()
    df["level"] = level[~bad_grade].astype(int)

    parsed = df["image"].map(parse_name)
    bad_name = parsed.isna()
    issues["unparseable_names"] = df.loc[bad_name, "image"].tolist()[:50]
    issues["unparseable_names_count"] = int(bad_name.sum())
    df = df[~bad_name].copy()
    df["patient_id"] = parsed[~bad_name].map(lambda t: t[0])
    df["eye"] = parsed[~bad_name].map(lambda t: t[1])

    has_img = df["image"].isin(index.keys())
    issues["missing_images"] = df.loc[~has_img, "image"].tolist()[:50]
    issues["missing_images_count"] = int((~has_img).sum())
    df = df[has_img].copy()
    issues["images_without_label"] = int(len(set(index) - set(raw["image"])))

    df["label"] = (df["level"] >= referable_min_grade).astype(int)
    df["path"] = df["image"].map(lambda s: str(index[s]))
    issues["rows_after_validation"] = len(df)
    return df.reset_index(drop=True), issues


# --------------------------------------------------------------------------- statistics
def _analyse_file(args: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    path, qcfg = args
    try:
        data = P.read_bytes(path)
    except OSError:
        data = b""
    return P.analyse_bytes(data, qcfg) | {"file_bytes": len(data)}


def compute_stats(
    paths: list[str], qcfg: Mapping[str, Any], workers: int, cache_file: Path | None = None
) -> pd.DataFrame:
    """Statistics for every path, in parallel, reusing cached rows for unchanged files.

    The cache key is (path, size, mtime) so re-runs only touch new or modified files; the quality
    flag is always recomputed from the stats, so changing a threshold needs no re-scan.
    """
    keys = pd.DataFrame({"path": paths})
    st = [os.stat(p) for p in paths]
    keys["size"] = [s.st_size for s in st]
    keys["mtime_ns"] = [s.st_mtime_ns for s in st]
    cached = pd.DataFrame()
    if cache_file is not None and cache_file.exists():
        cached = pd.read_parquet(cache_file)
        merged = keys.merge(cached, on=["path", "size", "mtime_ns"], how="left", indicator=True)
        hit = merged["_merge"] == "both"
        done = merged[hit].drop(columns="_merge")
        todo = keys[~hit.to_numpy()]
    else:
        done, todo = pd.DataFrame(), keys
    if len(todo):
        qd = dict(qcfg)
        t0 = time.time()
        jobs = [(p, qd) for p in todo["path"]]
        if workers > 1 and len(jobs) > 20:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_analyse_file, jobs, chunksize=64))
        else:
            results = [_analyse_file(j) for j in jobs]
        new = pd.concat([todo.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        print(f"  computed stats for {len(new)} images in {time.time() - t0:.0f}s ({len(done)} cached)")
        done = pd.concat([done, new], ignore_index=True) if len(done) else new
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            keep = pd.concat([cached, new], ignore_index=True).drop_duplicates(["path"], keep="last") if len(cached) else new
            keep.drop(columns=["quality"], errors="ignore").to_parquet(cache_file, index=False)
    # Quality always recomputed from current thresholds.
    done["quality"] = [
        P.quality_flag(row, qcfg, bool(row["readable"])) for row in done.to_dict("records")
    ]
    return done.drop(columns=["size", "mtime_ns"])


# --------------------------------------------------------------------------- pipeline
def build_manifest(cfg: Mapping[str, Any], workers: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run discovery, validation and statistics; return (manifest, quality report)."""
    raw_dir = resolve_path(cfg, "paths.raw_dir")
    processed = resolve_path(cfg, "paths.processed_dir")
    if not raw_dir.is_dir():
        raise DataError(f"{raw_dir} does not exist. Run `python -m dr.download` first.")
    labels_csv = find_labels_csv(raw_dir, get(cfg, "data.labels_csv"))
    wanted = set(pd.read_csv(labels_csv, usecols=["image"], dtype=str)["image"].str.strip())
    index, index_info = index_images(raw_dir, wanted, get(cfg, "data.exclude_dirs", []) or [])
    df, issues = load_and_validate(labels_csv, index, int(get(cfg, "data.referable_min_grade", 2)))
    if df.empty:
        raise DataError("No usable labelled images after validation; see data_quality.json.")

    qcfg = dict(get(cfg, "quality"))
    n_workers = workers or int(get(cfg, "data.num_workers", 4))
    stats = compute_stats(df["path"].tolist(), qcfg, n_workers, processed / STATS_CACHE)
    df = df.merge(stats, on="path", how="left")
    df["aspect_ratio"] = df["width"] / df["height"]
    df["quality_ok"] = df["quality"] == P.QUALITY_OK
    df["path"] = df["path"].map(lambda p: Path(p).relative_to(raw_dir).as_posix())
    df = df[MANIFEST_COLUMNS].sort_values(["patient_id", "eye"], key=lambda s: s.astype(str)).reset_index(drop=True)

    report = quality_report(df, issues, qcfg)
    report.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_dir": str(raw_dir),
            "labels_csv": labels_csv.relative_to(raw_dir).as_posix(),
            "index": index_info,
        }
    )
    return df, report


def quality_report(df: pd.DataFrame, issues: Mapping[str, Any], qcfg: Mapping[str, Any]) -> dict[str, Any]:
    """Summary numbers for data_quality.json (and the CLI printout)."""
    md5_groups = df.groupby("md5")["patient_id"].nunique()
    dup_md5 = md5_groups[md5_groups > 1]
    return {
        "issues": dict(issues),
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "grade_counts": {int(k): int(v) for k, v in df["level"].value_counts().sort_index().items()},
        "referable_share": round(float(df["label"].mean()), 4),
        "quality_counts": {k: int(v) for k, v in df["quality"].value_counts().items()},
        "poor_quality_share": round(float((~df["quality_ok"]).mean()), 4),
        "md5_duplicates": {
            "files_sharing_md5": int(df["md5"].duplicated(keep=False).sum()),
            "md5_groups_spanning_patients": int(len(dup_md5)),
            "examples": df[df["md5"].isin(dup_md5.index[:5])][["image", "md5"]].to_dict("records"),
        },
        "resolution_top10": {
            f"{int(w)}x{int(h)}": int(n)
            for (w, h), n in df.dropna(subset=["width"]).groupby(["width", "height"]).size().nlargest(10).items()
        },
        "thresholds": dict(qcfg),
    }


def write_outputs(df: pd.DataFrame, report: Mapping[str, Any], processed: Path) -> tuple[Path, Path]:
    processed.mkdir(parents=True, exist_ok=True)
    manifest_path = processed / MANIFEST
    df.to_csv(manifest_path, index=False)
    report_path = processed / QUALITY_REPORT
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return manifest_path, report_path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def print_summary(report: Mapping[str, Any]) -> None:
    iss = report["issues"]
    print(f"Labels CSV: {report.get('labels_csv')}  ({iss['label_rows']} rows)")
    print(
        f"Validation: {iss['duplicate_label_rows']} duplicate rows, {iss['invalid_grades_count']} invalid grades, "
        f"{iss['unparseable_names_count']} unparseable names, {iss['missing_images_count']} missing images, "
        f"{iss['images_without_label']} images without label"
    )
    print(f"Images: {report['n_images']}  patients: {report['n_patients']}  referable: {report['referable_share']:.2%}")
    print(f"Grades: {report['grade_counts']}")
    print(f"Quality: {report['quality_counts']}  (poor: {report['poor_quality_share']:.1%})")
    md = report["md5_duplicates"]
    print(f"MD5: {md['files_sharing_md5']} files share an MD5; {md['md5_groups_spanning_patients']} groups span patients")


# =========================================================================== Phase 3: split
SPLITS = ("train", "val", "test")
SPLIT_SUMMARY = "split_summary.json"
CACHE_INFO = "CACHE_INFO.json"


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Group id per image: same patient, or connected through a shared MD5, means same group.

    Why: both eyes of a patient are correlated, and a byte-identical file under two patient ids
    is the same photograph. Either one crossing the train/test boundary leaks information.
    Implemented as union-find over patient ids, joined by MD5.
    """
    parent: dict[str, str] = {p: p for p in df["patient_id"].astype(str).unique()}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, pids in df.dropna(subset=["md5"]).groupby("md5")["patient_id"]:
        ids = pids.astype(str).unique()
        for other in ids[1:]:
            ra, rb = find(ids[0]), find(other)
            if ra != rb:
                parent[max(ra, rb, key=_pid_key)] = min(ra, rb, key=_pid_key)
    return df["patient_id"].astype(str).map(lambda p: f"g{find(p)}")


def _pid_key(p: str) -> tuple[int, str]:
    return (int(p), p) if p.isdigit() else (sys.maxsize, p)


def _group_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group: image count and the worst-eye label used for stratification."""
    return (
        df.groupby("group_id")
        .agg(n_images=("image", "size"), group_label=("label", "max"))
        .reset_index()
        .sort_values("group_id", key=lambda s: s.map(lambda g: _pid_key(g[1:])))
        .reset_index(drop=True)
    )


def subset_groups(groups: pd.DataFrame, max_images: int | None, seed: int) -> set[str]:
    """Stratified *patient-group* subset with about ``max_images`` images (local_cpu profile)."""
    total = int(groups["n_images"].sum())
    if not max_images or max_images >= total:
        return set(groups["group_id"])
    frac = max_images / total
    keep: list[str] = []
    for cls, g in groups.groupby("group_label"):
        g = g.sample(frac=1.0, random_state=seed + int(cls))
        target = frac * g["n_images"].sum()
        cum = g["n_images"].cumsum()
        keep += g.loc[cum <= max(target, g["n_images"].iloc[0]), "group_id"].tolist()
    return set(keep)


def split_groups(groups: pd.DataFrame, fractions: Mapping[str, float], seed: int) -> dict[str, str]:
    """Stratified train/val/test assignment of groups by worst-eye label.

    Groups of each class are shuffled and laid out along their cumulative image count, so the
    image-level proportions match the requested fractions even though groups differ in size.
    Every split gets at least one group per class whenever the class has three or more groups.
    """
    f_train = float(fractions["train"])
    f_val = float(fractions["val"])
    if abs(f_train + f_val + float(fractions["test"]) - 1.0) > 1e-6:
        raise DataError(f"data.split fractions must sum to 1, got {dict(fractions)}")
    out: dict[str, str] = {}
    for cls, g in groups.groupby("group_label"):
        g = g.sample(frac=1.0, random_state=seed + 101 * (int(cls) + 1)).reset_index(drop=True)
        pos = (g["n_images"].cumsum() - g["n_images"] / 2) / g["n_images"].sum()
        assigned = np.where(pos < f_train, "train", np.where(pos < f_train + f_val, "val", "test"))
        if len(g) >= 3:
            for need in ("test", "val"):
                if need not in assigned:
                    idx = int(np.flatnonzero(assigned == "train")[-1])
                    assigned[idx] = need
        out.update(dict(zip(g["group_id"], assigned, strict=True)))
    return out


def make_splits(df: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add ``group_id``, ``group_label`` and ``split`` (train/val/test/unused/excluded) columns."""
    seed = int(get(cfg, "project.seed", 42))
    out = df.copy()
    excluded = out["quality"].isin(get(cfg, "data.exclude_quality", []) or [])
    out["group_id"] = assign_groups(out)
    usable = out[~excluded]
    groups = _group_table(usable)
    chosen = subset_groups(groups, get(cfg, "data.max_images"), seed)
    groups = groups[groups["group_id"].isin(chosen)]
    assignment = split_groups(groups, get(cfg, "data.split"), seed)
    out["group_label"] = out["group_id"].map(dict(zip(groups["group_id"], groups["group_label"], strict=True)))
    out["split"] = out["group_id"].map(assignment).fillna("unused")
    out.loc[excluded, "split"] = "excluded"

    summary = split_summary(out)
    summary.update(
        {
            "seed": seed,
            "max_images": get(cfg, "data.max_images"),
            "fractions": dict(get(cfg, "data.split")),
            "excluded_quality": list(get(cfg, "data.exclude_quality", []) or []),
            "profile": get(cfg, "meta.profile"),
        }
    )
    for s in SPLITS:
        if summary["splits"].get(s, {}).get("positives", 0) == 0 or summary["splits"].get(s, {}).get("negatives", 0) == 0:
            raise DataError(f"Split {s!r} lacks one of the classes; increase data.max_images. {summary['splits']}")
    return out, summary


def split_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Counts per split plus leakage checks (recorded in split_summary.json)."""
    used = df[df["split"].isin(SPLITS)]
    per: dict[str, Any] = {}
    for s in SPLITS:
        part = used[used["split"] == s]
        per[s] = {
            "images": int(len(part)),
            "patients": int(part["patient_id"].nunique()),
            "groups": int(part["group_id"].nunique()),
            "positives": int(part["label"].sum()),
            "negatives": int((part["label"] == 0).sum()),
            "referable_share": round(float(part["label"].mean()), 4) if len(part) else None,
            "poor_quality_share": round(float((~part["quality_ok"]).mean()), 4) if len(part) else None,
        }
    leaks = {
        "patients_in_multiple_splits": int((used.groupby("patient_id")["split"].nunique() > 1).sum()),
        "md5_in_multiple_splits": int((used.groupby("md5")["split"].nunique() > 1).sum()),
        "groups_in_multiple_splits": int((used.groupby("group_id")["split"].nunique() > 1).sum()),
    }
    return {
        "splits": per,
        "unused": int((df["split"] == "unused").sum()),
        "excluded": int((df["split"] == "excluded").sum()),
        "leakage": leaks,
        "merged_md5_groups": int((df.groupby("group_id")["patient_id"].nunique() > 1).sum()),
    }


# =========================================================================== Phase 3: cache
def cache_dir(cfg: Mapping[str, Any]) -> Path:
    size = int(get(cfg, "preprocess.image_size"))
    return resolve_path(cfg, "paths.processed_dir") / f"images_{size}"


def cache_path(cfg: Mapping[str, Any], image: str) -> Path:
    return cache_dir(cfg) / f"{image}.png"


def _cache_one(args: tuple[str, str, int, dict[str, Any], float]) -> str | None:
    src, dst, size, pre, thr = args
    img = P.read_image(src)
    if img is None:
        return src
    P.save_png(P.preprocess_image(img, size, pre, thr), dst)
    return None


def cache_preprocessed(df: pd.DataFrame, cfg: Mapping[str, Any], workers: int, force: bool = False) -> dict[str, Any]:
    """Preprocess every train/val/test image into ``images_<size>/<image>.png``.

    The directory carries the preprocessing signature; if the settings change, cached files are
    rebuilt rather than silently reused.
    """
    size = int(get(cfg, "preprocess.image_size"))
    pre = dict(get(cfg, "preprocess"))
    thr = float(get(cfg, "quality.mask_threshold", 10.0))
    out_dir = cache_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = P.preprocess_signature(size, pre, thr)
    info_file = out_dir / CACHE_INFO
    if info_file.exists() and json.loads(info_file.read_text(encoding="utf-8")).get("signature") != sig:
        print(f"  preprocessing settings changed; rebuilding {out_dir}")
        force = True
    raw_dir = resolve_path(cfg, "paths.raw_dir")
    rows = df[df["split"].isin(SPLITS)]
    jobs = [
        (str(raw_dir / r.path), str(out_dir / f"{r.image}.png"), size, pre, thr)
        for r in rows.itertuples()
        if force or not (out_dir / f"{r.image}.png").exists()
    ]
    t0 = time.time()
    if workers > 1 and len(jobs) > 20:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            failed = [f for f in ex.map(_cache_one, jobs, chunksize=32) if f]
    else:
        failed = [f for f in map(_cache_one, jobs) if f]
    info = {"signature": sig, "images": int(len(rows)), "written": len(jobs) - len(failed), "failed": failed[:20]}
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"  cached {len(jobs) - len(failed)} PNGs ({len(rows) - len(jobs)} already present) in {time.time() - t0:.0f}s -> {out_dir}")
    if failed:
        raise DataError(f"{len(failed)} images could not be preprocessed, e.g. {failed[:3]}")
    return info


def save_visual_check(df: pd.DataFrame, cfg: Mapping[str, Any], out_path: Path, n: int = 4, seed: int = 0) -> Path:
    """Side-by-side raw vs preprocessed images, one row per sample, for a visual sanity check."""
    import cv2

    raw_dir = resolve_path(cfg, "paths.raw_dir")
    size = int(get(cfg, "preprocess.image_size"))
    pool = df[df["split"].isin(SPLITS) & df["quality_ok"]]
    sample = pd.concat(
        [g.sample(min(len(g), max(1, n // 2)), random_state=seed) for _, g in pool.groupby("label")]
    )
    rows = []
    for r in sample.itertuples():
        raw = P.read_image(raw_dir / r.path)
        left = P._pad_to_square(P.resize_long_side(raw, size), 0)
        right = P.load_png(cache_path(cfg, r.image))[..., ::-1]
        row = np.hstack([left, np.full((size, 8, 3), 255, np.uint8), right])
        cv2.putText(row, f"{r.image} grade {r.level}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    P.save_png(np.vstack(rows)[..., ::-1], out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(argparse.ArgumentParser(description="Ingest, validate, split and preprocess the data."))
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: data.num_workers).")
    parser.add_argument("--skip-cache", action="store_true", help="Stop after the split; do not write PNGs.")
    parser.add_argument("--force-cache", action="store_true", help="Rebuild every cached PNG.")
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
        workers = args.workers or int(get(cfg, "data.num_workers", 4))
        df, report = build_manifest(cfg, workers)
        processed = resolve_path(cfg, "paths.processed_dir")
        df, summary = make_splits(df, cfg)
        manifest_path, report_path = write_outputs(df, report, processed)
        (processed / SPLIT_SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print_summary(report)
        print_split_summary(summary)
        if not args.skip_cache:
            cache_preprocessed(df, cfg, workers, force=args.force_cache)
            check = save_visual_check(df, cfg, resolve_path(cfg, "paths.reports_dir") / "local" / "preprocess_check.png")
            print(f"Visual check: {check}")
    except (DataError, ConfigError) as exc:
        print(f"Data step failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {manifest_path}, {report_path} and {processed / SPLIT_SUMMARY}")
    return 0


def print_split_summary(summary: Mapping[str, Any]) -> None:
    for s, v in summary["splits"].items():
        print(f"  {s:<5} {v['images']:>6} images  {v['patients']:>6} patients  referable {v['referable_share']:.2%}")
    print(f"  unused {summary['unused']}, excluded {summary['excluded']}, leakage {summary['leakage']}")


if __name__ == "__main__":
    raise SystemExit(main())
