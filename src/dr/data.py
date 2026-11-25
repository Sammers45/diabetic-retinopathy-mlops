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


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(argparse.ArgumentParser(description="Ingest, validate and profile the data."))
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: data.num_workers).")
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
        df, report = build_manifest(cfg, args.workers)
    except (DataError, ConfigError) as exc:
        print(f"Data step failed: {exc}", file=sys.stderr)
        return 2
    manifest_path, report_path = write_outputs(df, report, resolve_path(cfg, "paths.processed_dir"))
    print_summary(report)
    print(f"Wrote {manifest_path} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
