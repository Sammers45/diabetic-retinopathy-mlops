"""Phase 1: download module. kagglehub is replaced by a fake module; no network is used."""

from __future__ import annotations

import json
import sys
import types
import zipfile
from collections import namedtuple
from pathlib import Path

import pytest
import requests

from dr import download as D
from dr.config import load_config

FAKE_TOKEN = "KGAT_test_secret_value_never_print_me"


class FakeKaggleHub(types.ModuleType):
    """Stands in for kagglehub; records calls and writes fake files into output_dir."""

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__("kagglehub")
        self.calls: list[tuple[str, str, dict]] = []
        self.error = error

    def _write(self, output_dir: str, name: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if name.endswith(".zip"):
            with zipfile.ZipFile(out / name, "w") as zf:
                zf.writestr("trainLabels.csv", "image,level\n10_left,0\n10_right,2\n")
        else:
            (out / name).write_text("synthetic", encoding="utf-8")

    def dataset_download(self, handle: str, path: str | None = None, **kw) -> str:
        self.calls.append(("dataset", handle, {"path": path, **kw}))
        if self.error:
            raise self.error
        self._write(kw["output_dir"], "trainLabels.csv")
        img_dir = Path(kw["output_dir"]) / "resized_train" / "resized_train"
        img_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / "10_left.jpeg").write_bytes(b"\xff\xd8synthetic")
        self._write(kw["output_dir"], "extra.zip")
        return kw["output_dir"]

    def competition_download(self, handle: str, path: str | None = None, **kw) -> str:
        self.calls.append(("competition", handle, {"path": path, **kw}))
        if self.error:
            raise self.error
        self._write(kw["output_dir"], path or "train.zip")
        return kw["output_dir"]


def http_error(status: int) -> Exception:
    """Build an error shaped like kagglehub's KaggleApiHTTPError (a requests.HTTPError)."""
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(f"{status} Client Error", response=resp)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated credentials dir, raw dir and fake kagglehub."""
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)
    kdir = tmp_path / "kaggle_home"
    kdir.mkdir()
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(kdir))
    fake = FakeKaggleHub()
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    raw = tmp_path / "raw"

    def cfg(*overrides: str):
        return load_config("local_cpu", overrides=[f"paths.raw_dir={raw.as_posix()}", "data.min_free_gb=0", *overrides])

    return types.SimpleNamespace(kdir=kdir, fake=fake, raw=raw, cfg=cfg, monkeypatch=monkeypatch)


def with_token(env) -> None:
    (env.kdir / "access_token").write_text(FAKE_TOKEN, encoding="utf-8")


# ------------------------------------------------------------------ credentials
def test_no_credentials_gives_actionable_message_and_no_network(env) -> None:
    with pytest.raises(D.DownloadError) as err:
        D.download(env.cfg())
    msg = str(err.value)
    assert "access_token" in msg and "KAGGLE_API_TOKEN" in msg and "kaggle.json" in msg
    assert env.fake.calls == [], "must fail before calling kagglehub"


def test_misnamed_token_file_is_pointed_out(env) -> None:
    (env.kdir / "access_token.env").write_text("KAGGLE_TOKEN: x", encoding="utf-8")
    with pytest.raises(D.DownloadError, match="named exactly 'access_token'"):
        D.download(env.cfg())


@pytest.mark.parametrize(
    "setup, expected",
    [
        (lambda e: e.monkeypatch.setenv("KAGGLE_API_TOKEN", FAKE_TOKEN), "env:KAGGLE_API_TOKEN"),
        (lambda e: (e.kdir / "kaggle.json").write_text("{}", encoding="utf-8"), "file:kaggle.json"),
        (with_token, "file:access_token"),
    ],
)
def test_credential_sources_detected(env, setup, expected: str) -> None:
    setup(env)
    assert expected in D.credential_sources()


def test_token_never_printed(env, capsys: pytest.CaptureFixture[str]) -> None:
    env.monkeypatch.setenv("KAGGLE_API_TOKEN", FAKE_TOKEN)
    env.fake.error = http_error(401)
    code = D.main(["--profile", "local_cpu", "--set", f"paths.raw_dir={env.raw.as_posix()}", "--set", "data.min_free_gb=0"])
    out = capsys.readouterr()
    assert code == 2
    assert FAKE_TOKEN not in out.out + out.err
    assert "kaggle.com/settings" in out.err


# ------------------------------------------------------------------ kaggle_dataset
def test_dataset_download_success(env) -> None:
    with_token(env)
    result = D.download(env.cfg("data.kaggle_dataset=owner/eyepacs-resized"))
    target = env.raw / "kaggle"
    assert Path(result.target_dir) == target
    name, handle, kw = env.fake.calls[0]
    assert (name, handle) == ("dataset", "owner/eyepacs-resized")
    assert Path(kw["output_dir"]) == target, "downloads go straight into data/raw/kaggle"
    assert (target / "resized_train" / "resized_train" / "10_left.jpeg").exists()
    assert (target / "extra" / "trainLabels.csv").exists(), "zips are extracted"
    info = json.loads((target / D.INFO_FILE).read_text(encoding="utf-8"))
    assert info["source"] == "kaggle_dataset" and info["handle"] == "owner/eyepacs-resized"
    assert info["files"] >= 3 and "extra.zip" in info["extracted"]
    # Idempotent: second run does not re-extract.
    again = D.download(env.cfg("data.kaggle_dataset=owner/eyepacs-resized"))
    assert again.extracted == []


@pytest.mark.parametrize("handle", ["null", "<owner>/<dataset>", "justaname", "a/b/c"])
def test_dataset_handle_validated(env, handle: str) -> None:
    with_token(env)
    with pytest.raises(D.DownloadError, match="data.kaggle_dataset"):
        D.download(env.cfg(f"data.kaggle_dataset={handle}"))
    assert env.fake.calls == []


@pytest.mark.parametrize(
    "status, needle",
    [(401, "kaggle.com/settings"), (403, "accept any terms"), (404, "Check the handle")],
)
def test_dataset_http_errors_are_actionable(env, status: int, needle: str) -> None:
    with_token(env)
    env.fake.error = http_error(status)
    with pytest.raises(D.DownloadError, match=needle):
        D.download(env.cfg("data.kaggle_dataset=owner/missing"))


def test_network_error_message(env) -> None:
    with_token(env)
    env.fake.error = requests.ConnectionError("connection reset")
    with pytest.raises(D.DownloadError, match="internet connection"):
        D.download(env.cfg("data.kaggle_dataset=owner/x"))


# ------------------------------------------------------------------ kaggle_competition
def test_competition_light_download_by_default(env, capsys) -> None:
    with_token(env)
    D.download(env.cfg("data.source=kaggle_competition"))
    paths = [kw["path"] for _, _, kw in env.fake.calls]
    assert paths == list(D.COMPETITION_LIGHT_FILES)
    assert all(h == "diabetic-retinopathy-detection" for _, h, _ in env.fake.calls)
    assert (env.raw / "kaggle" / "trainLabels.csv" / "trainLabels.csv").exists()
    assert "--full-competition" in capsys.readouterr().out


def test_competition_rules_not_accepted(env) -> None:
    with_token(env)
    env.fake.error = http_error(403)
    with pytest.raises(D.DownloadError, match=r"competitions/diabetic-retinopathy-detection/rules"):
        D.download(env.cfg("data.source=kaggle_competition"))


def test_full_competition_requires_disk_space(env) -> None:
    with_token(env)
    Usage = namedtuple("Usage", "total used free")
    env.monkeypatch.setattr(D.shutil, "disk_usage", lambda p: Usage(100e9, 90e9, 10e9))
    with pytest.raises(D.DownloadError, match="Not enough disk space"):
        D.download(env.cfg("data.source=kaggle_competition"), full_competition=True)
    assert env.fake.calls == []


def test_full_competition_opt_in_downloads_everything(env) -> None:
    with_token(env)
    Usage = namedtuple("Usage", "total used free")
    env.monkeypatch.setattr(D.shutil, "disk_usage", lambda p: Usage(500e9, 100e9, 400e9))
    D.download(env.cfg("data.source=kaggle_competition", "data.allow_full_competition_download=true"))
    assert [kw["path"] for _, _, kw in env.fake.calls] == [None]


# ------------------------------------------------------------------ local + archives
def test_local_source(env) -> None:
    with pytest.raises(D.DownloadError, match="data.source=local"):
        D.download(env.cfg("data.source=local"))
    env.raw.mkdir(parents=True, exist_ok=True)
    (env.raw / "trainLabels.csv").write_text("image,level\n", encoding="utf-8")
    result = D.download(env.cfg("data.source=local"))
    assert result.files == 1 and env.fake.calls == []


def test_multipart_archives_are_joined(tmp_path: Path) -> None:
    whole = tmp_path / "build.zip"
    with zipfile.ZipFile(whole, "w") as zf:
        zf.writestr("train/1_left.jpeg", b"x" * 5000)
    data = whole.read_bytes()
    whole.unlink()
    root = tmp_path / "root"
    root.mkdir()
    third = len(data) // 3
    for i, chunk in enumerate([data[:third], data[third : 2 * third], data[2 * third :]], start=1):
        (root / f"train.zip.{i:03d}").write_bytes(chunk)
    extracted = D.extract_archives(root)
    assert extracted == ["train.zip"]
    assert (root / "train" / "train" / "1_left.jpeg").read_bytes() == b"x" * 5000


def test_zip_slip_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../../escaped.txt", "nope")
    with pytest.raises(D.DownloadError, match="unsafe path"):
        D.extract_archives(tmp_path)
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_corrupt_zip_is_actionable(tmp_path: Path) -> None:
    (tmp_path / "broken.zip").write_bytes(b"PK\x03\x04garbage")
    with pytest.raises(D.DownloadError, match="--force"):
        D.extract_archives(tmp_path)


def test_unknown_source(env) -> None:
    with pytest.raises(D.DownloadError, match="expected one of"):
        D.download(env.cfg("data.source=ftp"))
