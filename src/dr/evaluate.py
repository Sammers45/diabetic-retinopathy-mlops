"""Threshold selection, test metrics with bootstrap CIs, subgroup metrics and plots.

Rules:

* The operating threshold is chosen on **validation** data only: the highest threshold that still
  gives at least ``evaluate.target_sensitivity`` (0.90). A higher threshold means fewer false
  positives, so it is the best specificity available at that sensitivity.
* The **test** set is scored exactly once per training run, with that fixed threshold.
* Confidence intervals come from a *patient-level* (cluster) bootstrap. Both eyes of a patient
  are correlated, so resampling single images would make the intervals too narrow.

Plots use ``matplotlib.figure.Figure`` directly (no pyplot), so they work headless.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

METRICS = ("roc_auc", "pr_auc", "sensitivity", "specificity", "ppv", "npv", "accuracy", "brier")


# --------------------------------------------------------------------------- threshold
def choose_threshold(y: np.ndarray, p: np.ndarray, target_sensitivity: float = 0.90) -> float:
    """Highest threshold ``t`` with sensitivity(p >= t) >= target on the given (validation) data."""
    y = np.asarray(y).astype(int)
    pos = np.sort(np.asarray(p, dtype=float)[y == 1])[::-1]
    if len(pos) == 0:
        raise ValueError("Cannot choose a threshold: no positive cases in the validation set.")
    k = math.ceil(target_sensitivity * len(pos) - 1e-9)
    return float(pos[max(k, 1) - 1])


# --------------------------------------------------------------------------- metrics
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """All operating-point and ranking metrics at ``threshold``."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "accuracy": _safe_div(tp + tn, len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
    }


def bootstrap_ci(
    df: pd.DataFrame, threshold: float, n_boot: int = 1000, seed: int = 0, cluster: str = "patient_id"
) -> dict[str, tuple[float, float]]:
    """95% percentile intervals from a cluster bootstrap over ``cluster`` (patients)."""
    rng = np.random.default_rng(seed)
    groups = df.groupby(cluster).indices
    keys = np.array(list(groups))
    idx_lists = [groups[k] for k in keys]
    y, p = df["label"].to_numpy(), df["p_referable"].to_numpy()
    samples: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([idx_lists[i] for i in pick])
        m = binary_metrics(y[idx], p[idx], threshold)
        for k in METRICS:
            samples[k].append(m[k])
    out = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        out[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) else (float("nan"),) * 2
    return out


def aspect_group(aspect_ratio: pd.Series) -> pd.Series:
    """Camera proxy used in the EDA: square, 4:3 or 3:2 frames."""
    return pd.cut(aspect_ratio, [0, 1.2, 1.42, 10], labels=["square", "4:3", "3:2"]).astype(str)


def subgroup_metrics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Metrics per eye, camera group (aspect ratio) and quality flag (point estimates + n)."""
    rows = []
    groupings = {
        "eye": df["eye"],
        "camera_aspect": aspect_group(df["aspect_ratio"]) if "aspect_ratio" in df else None,
        "quality": np.where(df["quality"] == "ok", "ok", "poor"),
    }
    for name, key in groupings.items():
        if key is None:
            continue
        for value, part in df.groupby(np.asarray(key)):
            m = binary_metrics(part["label"], part["p_referable"], threshold)
            rows.append({"subgroup": name, "value": value, **{k: m[k] for k in ("n", "prevalence", *METRICS)}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plots
def plot_roc(y: np.ndarray, p: np.ndarray, threshold: float, title: str) -> Figure:
    fpr, tpr, thr = roc_curve(y, p)
    op = binary_metrics(y, p, threshold)
    fig = Figure(figsize=(4.8, 4.4))
    ax = fig.subplots()
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC {op['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1, ls="--")
    ax.scatter([1 - op["specificity"]], [op["sensitivity"]], color="#c2410c", zorder=3, s=50,
               label=f"t={threshold:.3f}: sens {op['sensitivity']:.2f}, spec {op['specificity']:.2f}")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Figure:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    mean_p = [p[idx == b].mean() for b in range(bins) if (idx == b).any()]
    frac = [y[idx == b].mean() for b in range(bins) if (idx == b).any()]
    counts = [(idx == b).sum() for b in range(bins) if (idx == b).any()]
    fig = Figure(figsize=(4.8, 4.4))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1, ls="--", label="perfect")
    ax.plot(mean_p, frac, marker="o", color="#2563eb", label="model")
    for x, f, n in zip(mean_p, frac, counts, strict=True):
        ax.annotate(str(n), (x, f), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlabel("mean predicted P(referable)")
    ax.set_ylabel("observed referable fraction")
    ax.set_title(f"Calibration (Brier {np.mean((p - y) ** 2):.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_confusion(m: Mapping[str, float], threshold: float) -> Figure:
    mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig = Figure(figsize=(4.0, 3.6))
    ax = fig.subplots()
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > mat.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["not referable", "referable"])
    ax.set_yticks([0, 1], ["not referable", "referable"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Test confusion matrix (t={threshold:.3f})")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- orchestration
def evaluate(
    val: pd.DataFrame, test: pd.DataFrame, cfg: Mapping[str, Any], out_dir: Path
) -> dict[str, Any]:
    """Pick the threshold on ``val``, score ``test`` once, write plots/CSVs into ``out_dir``.

    ``val`` and ``test`` need ``label``, ``patient_id`` and ``p_referable`` columns.
    Returns a dict with the threshold, metrics, CIs and the paths written.
    """
    ecfg = cfg.get("evaluate", {})
    target = float(ecfg.get("target_sensitivity", 0.90))
    seed = int(cfg.get("project", {}).get("seed", 42))
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = choose_threshold(val["label"].to_numpy(), val["p_referable"].to_numpy(), target)
    val_m = binary_metrics(val["label"], val["p_referable"], threshold)
    test_m = binary_metrics(test["label"], test["p_referable"], threshold)
    ci = bootstrap_ci(test, threshold, int(ecfg.get("n_bootstrap", 1000)), seed)
    subgroups = subgroup_metrics(test, threshold)

    preds = test.copy()
    preds["threshold"] = threshold
    preds["prediction"] = (preds["p_referable"] >= threshold).astype(int)
    keep = [c for c in ("image", "patient_id", "eye", "level", "label", "p_referable", "prediction", "threshold", "quality", "aspect_ratio") if c in preds]
    preds[keep].to_csv(out_dir / "test_predictions.csv", index=False)
    subgroups.to_csv(out_dir / "subgroup_metrics.csv", index=False)

    y, p = test["label"].to_numpy(), test["p_referable"].to_numpy()
    plots = {
        "roc_test.png": plot_roc(y, p, threshold, "Test ROC (threshold from validation)"),
        "calibration_test.png": plot_calibration(y, p),
        "confusion_test.png": plot_confusion(test_m, threshold),
    }
    for name, fig in plots.items():
        fig.savefig(out_dir / name, dpi=120)

    summary = {
        "threshold": threshold,
        "target_sensitivity": target,
        "val": val_m,
        "test": test_m,
        "test_ci95": {k: list(v) for k, v in ci.items()},
        "bootstrap": {"n": int(ecfg.get("n_bootstrap", 1000)), "cluster": "patient_id", "seed": seed},
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary | {"subgroups": subgroups}


def mlflow_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    """Flatten an ``evaluate`` summary into MLflow metric names (test_*, val_*_at_threshold, CIs)."""
    out = {"threshold": summary["threshold"]}
    for k in (*METRICS, "tp", "fp", "tn", "fn"):
        out[f"test_{k}"] = float(summary["test"][k])
    for k in ("sensitivity", "specificity", "roc_auc"):
        out[f"val_{k}_at_threshold" if k != "roc_auc" else "val_roc_auc"] = float(summary["val"][k])
    for k, (lo, hi) in summary["test_ci95"].items():
        out[f"test_{k}_ci_low"] = lo
        out[f"test_{k}_ci_high"] = hi
    return {k: v for k, v in out.items() if not (isinstance(v, float) and math.isnan(v))}
