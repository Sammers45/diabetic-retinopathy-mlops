"""Append-only prediction log, written by the API and the production replay; read by the monitor.

Each row is one scored image: when it was scored, by which model version, the score and decision,
the image statistics and quality flag, the PCA embedding ``emb_*`` (drift features), and,
once ``/feedback`` arrives, the clinician's grade and the derived label.

Why SQLite rather than Parquet: feedback arrives later and updates a row in place. SQLite does
that in one statement, while Parquet would need the whole file rewritten for every label. One
file is also easy to copy, inspect and back up. A new connection is opened for every operation,
because FastAPI runs sync endpoints in a thread pool and a ``sqlite3`` connection must not be
shared across threads.

Images are never stored here. ``image_path`` is only filled when serving runs with
``log_images: true``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dr.preprocess import STAT_COLUMNS

TABLE = "predictions"

# Fixed columns; ``emb_0..emb_{k-1}`` are added on demand, because k depends on the model.
BASE_COLUMNS: dict[str, str] = {
    "request_id": "TEXT PRIMARY KEY",
    "timestamp": "TEXT NOT NULL",  # UTC, ISO 8601: sorts correctly as text
    "source": "TEXT",  # api | simulate | ...
    "model_name": "TEXT",
    "model_version": "TEXT",
    "p_referable": "REAL",
    "prediction": "INTEGER",
    "threshold": "REAL",
    "quality": "TEXT",
    **{c: "REAL" for c in STAT_COLUMNS},
    "md5": "TEXT",  # lets retraining de-duplicate production images against the test set
    "patient_id": "TEXT",
    "eye": "TEXT",
    "filename": "TEXT",
    "image_path": "TEXT",
    "error": "TEXT",
    "grade": "INTEGER",
    "label": "INTEGER",
    "feedback_at": "TEXT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clean(value: Any) -> Any:
    """numpy/pandas scalars and missing values -> plain Python values SQLite accepts."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    return value


class PredictionLog:
    """A thin wrapper around one SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            cols = ", ".join(f'"{c}" {t}' for c, t in BASE_COLUMNS.items())
            con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ({cols})")
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_timestamp ON {TABLE}(timestamp)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        try:
            with con:  # commits on success, rolls back on error
                yield con
        finally:
            con.close()

    def columns(self) -> list[str]:
        with self._connect() as con:
            return [r[1] for r in con.execute(f"PRAGMA table_info({TABLE})")]

    def _ensure_columns(self, con: sqlite3.Connection, names: Iterable[str]) -> None:
        existing = {r[1] for r in con.execute(f"PRAGMA table_info({TABLE})")}
        for name in names:
            if name not in existing:
                if not name.startswith("emb_") or not name[4:].isdigit():
                    raise ValueError(f"unknown prediction-log column: {name!r}")
                con.execute(f'ALTER TABLE {TABLE} ADD COLUMN "{name}" REAL')

    def append(self, records: Sequence[Mapping[str, Any]]) -> int:
        """Insert rows. Each record needs ``request_id``; ``timestamp`` defaults to now (UTC)."""
        if not records:
            return 0
        rows = [{"timestamp": utc_now(), **r} for r in records]
        names = sorted({k for r in rows for k in r}, key=lambda c: (c.startswith("emb_"), c))
        with self._connect() as con:
            self._ensure_columns(con, names)
            placeholders = ", ".join("?" for _ in names)
            quoted = ", ".join(f'"{n}"' for n in names)
            con.executemany(
                f"INSERT INTO {TABLE} ({quoted}) VALUES ({placeholders})",
                [tuple(_clean(r.get(n)) for n in names) for r in rows],
            )
        return len(rows)

    def add_feedback(self, request_id: str, grade: int, referable_min_grade: int = 2) -> dict[str, Any] | None:
        """Attach a clinician's ICDR grade to a logged prediction.

        Returns the updated row, or ``None`` if the request ID is unknown. A second call
        overwrites the first, so a mistaken grade can be corrected.
        """
        label = int(grade >= referable_min_grade)
        with self._connect() as con:
            cur = con.execute(
                f"UPDATE {TABLE} SET grade = ?, label = ?, feedback_at = ? WHERE request_id = ?",
                (int(grade), label, utc_now(), request_id),
            )
            if cur.rowcount == 0:
                return None
        return self.get(request_id)

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            row = con.execute(f"SELECT * FROM {TABLE} WHERE request_id = ?", (request_id,)).fetchone()
        return None if row is None else dict(row)

    def read(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Rows with ``start <= timestamp < end`` (ISO strings, either bound optional), oldest first."""
        where, args = [], []
        if start is not None:
            where.append("timestamp >= ?")
            args.append(start)
        if end is not None:
            where.append("timestamp < ?")
            args.append(end)
        sql = f"SELECT * FROM {TABLE}" + (f" WHERE {' AND '.join(where)}" if where else "") + " ORDER BY timestamp, rowid"
        with self._connect() as con:
            return pd.read_sql_query(sql, con, params=args)

    def __len__(self) -> int:
        with self._connect() as con:
            return int(con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])
