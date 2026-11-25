"""Fetch the EyePACS data into ``data/raw/`` (``python -m dr.download``).

``data.source`` selects how:

* ``local``: the data is already in ``data/raw/``; just verify it is there.
* ``kaggle_dataset``: ``kagglehub.dataset_download(handle)``, e.g. a resized EyePACS copy.
* ``kaggle_competition``: the original 2015 competition. By default only the labels and the sample
  archive are fetched; the full ~35 GB archive needs ``data.allow_full_competition_download=true``
  (or ``--full-competition``) and passes a disk-space check first.

Kaggle downloads go straight to ``data/raw/kaggle`` via kagglehub's ``output_dir`` option rather
than into the kagglehub cache. Why: a cache plus a copy would need twice the disk space, and a
symlink into the cache breaks on Windows without developer mode.

Failures (no token, rules not accepted, wrong handle, no disk space) raise ``DownloadError`` with
a message that says exactly what to do. Credentials are only checked for existence; the token is
never read into our code or printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dr.config import ConfigError, add_config_args, config_from_args, get, resolve_path

SOURCES = ("local", "kaggle_dataset", "kaggle_competition")
KAGGLE_SUBDIR = "kaggle"
INFO_FILE = "DOWNLOAD_INFO.json"
# Competition files fetched without the full-download opt-in.
COMPETITION_LIGHT_FILES = ("trainLabels.csv.zip", "sample.zip")
FULL_COMPETITION_GB = 35.0
# Archive + extracted copy, with headroom.
FULL_COMPETITION_REQUIRED_GB = 2.2 * FULL_COMPETITION_GB

NO_CREDENTIALS_HELP = """\
No Kaggle credentials found. Provide ONE of these, then re-run:
  1. A token file at {config_dir}{sep}access_token
     (Kaggle -> Settings -> API -> "Generate New Token"; the file holds only the token,
     no extension, no "KAGGLE_TOKEN:" label)
  2. The environment variable KAGGLE_API_TOKEN set to that token
  3. A legacy {config_dir}{sep}kaggle.json ({{"username": ..., "key": ...}})
In Google Colab, add a secret named KAGGLE_API_TOKEN (key icon in the sidebar).
Never commit the token; data/, .env and kaggle.json are git-ignored."""


class DownloadError(RuntimeError):
    """A download problem with an actionable message for the user."""


@dataclass
class DownloadResult:
    """Where the data landed and how it got there (also written to DOWNLOAD_INFO.json)."""

    source: str
    raw_dir: str
    target_dir: str
    handle: str | None = None
    files: int = 0
    bytes: int = 0
    extracted: list[str] = field(default_factory=list)
    downloaded_at: str = ""
    kagglehub_version: str | None = None


# --------------------------------------------------------------------------- credentials
def kaggle_config_dir() -> Path:
    """Directory holding Kaggle credentials (``KAGGLE_CONFIG_DIR`` or ``~/.kaggle``)."""
    env = os.environ.get("KAGGLE_CONFIG_DIR")
    return Path(env).expanduser() if env else Path.home() / ".kaggle"


def credential_sources() -> list[str]:
    """Names of the credential sources that exist. Checks existence only, never contents."""
    found = []
    if os.environ.get("KAGGLE_API_TOKEN"):
        found.append("env:KAGGLE_API_TOKEN")
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        found.append("env:KAGGLE_USERNAME/KAGGLE_KEY")
    cdir = kaggle_config_dir()
    for name in ("access_token", "kaggle.json", "credentials.json"):
        if (cdir / name).is_file():
            found.append(f"file:{name}")
    return found


def require_credentials() -> list[str]:
    """Fail fast, before any network call, when no credentials are configured."""
    found = credential_sources()
    if found:
        return found
    cdir = kaggle_config_dir()
    hint = ""
    near_misses = sorted(p.name for p in cdir.glob("access_token*") if p.name != "access_token") if cdir.is_dir() else []
    if near_misses:
        hint = (
            f"\nFound {', '.join(near_misses)} in {cdir}: the file must be named exactly "
            "'access_token' (no extension) and contain only the token."
        )
    raise DownloadError(NO_CREDENTIALS_HELP.format(config_dir=cdir, sep=os.sep) + hint)


# --------------------------------------------------------------------------- helpers
def check_disk_space(path: Path, required_gb: float) -> float:
    """Raise if the drive holding ``path`` has less than ``required_gb`` free; return free GB."""
    probe = path
    while not probe.exists():
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < required_gb:
        raise DownloadError(
            f"Not enough disk space at {probe}: {free_gb:.1f} GB free, {required_gb:.1f} GB needed. "
            "Free some space, or point paths.raw_dir at a larger drive with --set paths.raw_dir=..."
        )
    return free_gb


def _merge_split_archives(root: Path) -> list[Path]:
    """Join multi-part archives (``train.zip.001``, ``.002``...) into ``train.zip``.

    The Kaggle competition ships the 35 GB training set as numbered parts.
    """
    merged: list[Path] = []
    for first in sorted(root.rglob("*.zip.001")):
        target = first.with_suffix("")  # strip ".001"
        if target.exists():
            continue
        parts = sorted(first.parent.glob(target.name + ".[0-9][0-9][0-9]"))
        with target.open("wb") as out:
            for part in parts:
                with part.open("rb") as src:
                    shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
        merged.append(target)
    return merged


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip, refusing members that would escape ``dest`` (zip-slip)."""
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest_resolved):
                raise DownloadError(f"Refusing to extract {archive.name}: unsafe path {member.filename!r}.")
        zf.extractall(dest)


def extract_archives(root: Path, delete: bool = False) -> list[str]:
    """Extract every ``.zip`` under ``root`` next to itself; idempotent via a marker file."""
    _merge_split_archives(root)
    done: list[str] = []
    for archive in sorted(root.rglob("*.zip")):
        dest = archive.with_suffix("")
        marker = dest / ".extracted"
        if marker.exists():
            continue
        try:
            dest.mkdir(parents=True, exist_ok=True)
            _safe_extract(archive, dest)
        except zipfile.BadZipFile as exc:
            raise DownloadError(
                f"{archive} is not a valid zip ({exc}). Delete it and re-run with --force."
            ) from exc
        marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        done.append(str(archive.relative_to(root)))
        if delete:
            archive.unlink()
    return done


def _tree_stats(root: Path) -> tuple[int, int]:
    files = [p for p in root.rglob("*") if p.is_file() and p.name != INFO_FILE]
    return len(files), sum(p.stat().st_size for p in files)


def _kaggle_error(exc: Exception, kind: str, handle: str) -> DownloadError:
    """Translate kagglehub/HTTP errors into actionable messages."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    name = type(exc).__name__
    if name in {"UnauthenticatedError", "CredentialError"} or status == 401:
        return DownloadError(
            f"Kaggle rejected the credentials ({name}{f', HTTP {status}' if status else ''}). "
            "The token may be expired, revoked or malformed. Generate a new one at "
            "https://www.kaggle.com/settings (API section) and save it as described:\n"
            + NO_CREDENTIALS_HELP.format(config_dir=kaggle_config_dir(), sep=os.sep)
        )
    if status == 403:
        if kind == "competition":
            return DownloadError(
                f"HTTP 403 for competition '{handle}'. You must accept the competition rules first: "
                f"open https://www.kaggle.com/competitions/{handle}/rules, click 'I Understand and "
                "Accept', then re-run."
            )
        return DownloadError(
            f"HTTP 403 for dataset '{handle}'. The dataset may be private or require consent on "
            f"https://www.kaggle.com/datasets/{handle}. Open it in a browser while logged in, accept "
            "any terms, then re-run."
        )
    if status == 404 or name == "NotFoundError":
        where = "competitions" if kind == "competition" else "datasets"
        return DownloadError(
            f"Kaggle {kind} '{handle}' was not found (HTTP 404). Check the handle; it must look like "
            f"'owner/dataset-name' and exist at https://www.kaggle.com/{where}/{handle}. Set it with "
            "--set data.kaggle_dataset=owner/dataset-name"
        )
    return DownloadError(
        f"Kaggle download failed ({name}{f', HTTP {status}' if status else ''}): {exc}. "
        "Check your internet connection and re-run; completed files are reused."
    )


# --------------------------------------------------------------------------- sources
def _download_local(raw_dir: Path) -> DownloadResult:
    if not raw_dir.is_dir() or not any(raw_dir.iterdir()):
        raise DownloadError(
            f"data.source=local but {raw_dir} is empty or missing. Copy the EyePACS labels CSV and "
            "images there, or use --set data.source=kaggle_dataset."
        )
    extracted = extract_archives(raw_dir)
    files, size = _tree_stats(raw_dir)
    return DownloadResult("local", str(raw_dir), str(raw_dir), None, files, size, extracted)


def _download_dataset(cfg: dict[str, Any], raw_dir: Path, force: bool) -> DownloadResult:
    handle = get(cfg, "data.kaggle_dataset")
    if not handle or "<" in str(handle) or str(handle).count("/") != 1:
        raise DownloadError(
            f"data.kaggle_dataset is {handle!r}; it must be a Kaggle handle 'owner/dataset-name' "
            "of an EyePACS copy with a labels CSV (image, level). Set it with "
            "--set data.kaggle_dataset=owner/dataset-name"
        )
    require_credentials()
    target = raw_dir / KAGGLE_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    check_disk_space(target, float(get(cfg, "data.min_free_gb", 20)))
    import kagglehub

    try:
        kagglehub.dataset_download(handle, force_download=force, output_dir=str(target))
    except Exception as exc:  # kagglehub raises several unrelated exception types
        raise _kaggle_error(exc, "dataset", handle) from exc
    return _finish("kaggle_dataset", raw_dir, target, handle, cfg)


def _download_competition(
    cfg: dict[str, Any], raw_dir: Path, force: bool, full: bool
) -> DownloadResult:
    handle = str(get(cfg, "data.kaggle_competition"))
    allow_full = full or bool(get(cfg, "data.allow_full_competition_download", False))
    require_credentials()
    target = raw_dir / KAGGLE_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    required = FULL_COMPETITION_REQUIRED_GB if allow_full else float(get(cfg, "data.min_free_gb", 20))
    check_disk_space(target, required)
    import kagglehub

    try:
        if allow_full:
            kagglehub.competition_download(handle, force_download=force, output_dir=str(target))
        else:
            for name in COMPETITION_LIGHT_FILES:
                kagglehub.competition_download(
                    handle, path=name, force_download=force, output_dir=str(target)
                )
    except Exception as exc:
        raise _kaggle_error(exc, "competition", handle) from exc
    result = _finish("kaggle_competition", raw_dir, target, handle, cfg)
    if not allow_full:
        print(
            "Fetched only the labels and sample images. The full ~35 GB training archive needs "
            "--full-competition (or --set data.allow_full_competition_download=true)."
        )
    return result


def _finish(source: str, raw_dir: Path, target: Path, handle: str, cfg: dict[str, Any]) -> DownloadResult:
    extracted = extract_archives(target, delete=bool(get(cfg, "data.delete_archives", False)))
    files, size = _tree_stats(target)
    if files == 0:
        raise DownloadError(f"Download finished but {target} is empty. Re-run with --force.")
    try:
        from importlib.metadata import version

        khv = version("kagglehub")
    except Exception:  # pragma: no cover
        khv = None
    return DownloadResult(source, str(raw_dir), str(target), handle, files, size, extracted, kagglehub_version=khv)


def download(cfg: dict[str, Any], force: bool = False, full_competition: bool = False) -> DownloadResult:
    """Run the configured download and write ``DOWNLOAD_INFO.json`` into the target directory."""
    source = get(cfg, "data.source")
    if source not in SOURCES:
        raise DownloadError(f"data.source={source!r}; expected one of {SOURCES}.")
    raw_dir = resolve_path(cfg, "paths.raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if source == "local":
        result = _download_local(raw_dir)
    elif source == "kaggle_dataset":
        result = _download_dataset(cfg, raw_dir, force)
    else:
        result = _download_competition(cfg, raw_dir, force, full_competition)
    result.downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(result.target_dir, INFO_FILE).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = add_config_args(argparse.ArgumentParser(description="Download EyePACS data into data/raw."))
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    parser.add_argument(
        "--full-competition",
        action="store_true",
        help="With data.source=kaggle_competition: download the full ~35 GB archive.",
    )
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
        result = download(cfg, force=args.force, full_competition=args.full_competition)
    except (DownloadError, ConfigError) as exc:
        print(f"\nDownload failed:\n{exc}", file=sys.stderr)
        return 2
    print(
        f"OK: {result.files} files ({result.bytes / 1e9:.2f} GB) in {result.target_dir} "
        f"[source={result.source}{', handle=' + result.handle if result.handle else ''}]"
    )
    if result.extracted:
        print(f"Extracted: {', '.join(result.extracted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
