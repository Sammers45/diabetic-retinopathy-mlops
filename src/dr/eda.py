"""Exploratory plots for the manifest; ``notebooks/01_eda.ipynb`` only calls these functions.

Why a module instead of notebook code: the plots are reused in reports and tested in CI on
synthetic data. Aggregate charts (counts, histograms) are saved to ``reports/eda/`` and may be
committed. Grids of example photographs are only *shown*, never saved to the repo, because the
EyePACS images are licensed and must not be redistributed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dr import preprocess as P  # noqa: E402

GRADE_NAMES = {0: "0 none", 1: "1 mild", 2: "2 moderate", 3: "3 severe", 4: "4 proliferative"}
REFERABLE = "#c2410c"
NON_REFERABLE = "#2563eb"


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"patient_id": str, "image": str})


def plot_grades(df: pd.DataFrame) -> plt.Figure:
    """Grade counts, coloured by the binary referable label."""
    counts = df["level"].value_counts().reindex(range(5), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = [NON_REFERABLE if g < 2 else REFERABLE for g in counts.index]
    bars = ax.bar([GRADE_NAMES[g] for g in counts.index], counts.values, color=colors)
    ax.bar_label(bars, labels=[f"{v:,}\n{v / counts.sum():.1%}" for v in counts.values], fontsize=8)
    ax.set_ylabel("images")
    ax.set_title(f"ICDR grade distribution (referable = grade ≥ 2: {df['label'].mean():.1%})")
    ax.margins(y=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_quality(df: pd.DataFrame) -> plt.Figure:
    """Quality flags overall and their share per grade."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    counts = df["quality"].value_counts()
    bars = axes[0].barh(counts.index[::-1], counts.values[::-1], color="#64748b")
    axes[0].bar_label(bars, labels=[f"{v:,} ({v / len(df):.1%})" for v in counts.values[::-1]], fontsize=8)
    axes[0].set_title("Quality flags")
    axes[0].margins(x=0.3)
    poor_by_grade = df.groupby("level")["quality_ok"].apply(lambda s: 1 - s.mean()).reindex(range(5))
    axes[1].bar([GRADE_NAMES[g] for g in range(5)], poor_by_grade.values, color="#64748b")
    axes[1].set_ylabel("share poor quality")
    axes[1].set_title("Poor-quality share by grade")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_stat_histograms(df: pd.DataFrame) -> plt.Figure:
    """Distributions of the image statistics used for quality flags and drift monitoring."""
    cols = ["brightness", "contrast", "sharpness", "fundus_fraction", "aspect_ratio", "height"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    ok = df[df["quality_ok"]]
    poor = df[~df["quality_ok"] & df["readable"]]
    for ax, col in zip(axes.ravel(), cols, strict=True):
        data = df[col].dropna()
        log = col == "sharpness"
        lo, hi = (np.log10(data.clip(lower=1e-2)).quantile([0.001, 0.999]) if log else data.quantile([0.001, 0.999]))
        if hi <= lo:  # constant column (e.g. every image resized to the same width)
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.logspace(lo, hi, 50) if log else np.linspace(lo, hi, 50)
        ax.hist(ok[col].dropna(), bins=bins, alpha=0.7, label="ok", color=NON_REFERABLE)
        ax.hist(poor[col].dropna(), bins=bins, alpha=0.7, label="poor", color=REFERABLE)
        if log:
            ax.set_xscale("log")
        ax.set_title(col)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend()
    fig.suptitle("Image statistics (computed inside the fundus mask at 512 px)")
    fig.tight_layout()
    return fig


def plot_resolutions(df: pd.DataFrame, top: int = 12) -> plt.Figure:
    """Resolution groups: a rough proxy for camera type (aspect ratio survives resizing)."""
    res = (df["width"].astype("Int64").astype(str) + "×" + df["height"].astype("Int64").astype(str)).value_counts()
    head = res.head(top)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(head.index[::-1], head.values[::-1], color="#64748b")
    ax.set_xlabel("images")
    ax.set_title(f"Top {len(head)} resolutions ({res.size} distinct)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_referable_by_group(df: pd.DataFrame) -> plt.Figure:
    """Referable rate by eye and by aspect-ratio bucket (subgroup sanity check)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    eye = df.groupby("eye")["label"].mean()
    axes[0].bar(eye.index, eye.values, color=REFERABLE)
    axes[0].set_title("Referable rate by eye")
    buckets = pd.cut(df["aspect_ratio"], [0, 1.05, 1.4, 1.6, 10], labels=["≈1:1", "≤1.4", "≤1.6", ">1.6"])
    grp = df.groupby(buckets, observed=True)["label"].agg(["mean", "size"])
    axes[1].bar(grp.index.astype(str), grp["mean"], color=REFERABLE)
    for i, (rate, n) in enumerate(zip(grp["mean"], grp["size"], strict=True)):
        axes[1].text(i, rate, f"{rate:.1%}\nn={n:,}", ha="center", va="bottom", fontsize=8)
    axes[1].margins(y=0.25)
    axes[1].set_title("Referable rate by aspect ratio (camera proxy)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def show_examples(
    df: pd.DataFrame, raw_dir: str | Path, by: str = "level", n: int = 4, seed: int = 0, size: int = 256
) -> plt.Figure:
    """Grid of example images per value of ``by`` (e.g. grade or quality). Display only."""
    raw_dir = Path(raw_dir)
    groups = sorted(df[by].dropna().unique(), key=str)
    fig, axes = plt.subplots(len(groups), n, figsize=(2.2 * n, 2.2 * len(groups)), squeeze=False)
    for r, g in enumerate(groups):
        sub = df[df[by] == g]
        sample = sub.sample(min(n, len(sub)), random_state=seed)
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(sample):
                continue
            row = sample.iloc[c]
            img = P.read_image(raw_dir / row["path"])
            if img is None:
                ax.text(0.5, 0.5, "unreadable", ha="center", va="center")
            else:
                ax.imshow(P.resize_long_side(img, size)[..., ::-1])
            ax.set_title(f"{row['image']}", fontsize=7)
        axes[r, 0].text(-0.1, 0.5, str(g), transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.tight_layout()
    return fig


AGGREGATE_PLOTS = {
    "grades": plot_grades,
    "quality": plot_quality,
    "stat_histograms": plot_stat_histograms,
    "resolutions": plot_resolutions,
    "referable_by_group": plot_referable_by_group,
}


def main(argv: list[str] | None = None) -> int:
    """``python -m dr.eda``: write the aggregate charts for the current manifest to reports/eda."""
    import argparse

    from dr.config import add_config_args, config_from_args, resolve_path

    parser = add_config_args(argparse.ArgumentParser(description="Save aggregate EDA charts."))
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    manifest = resolve_path(cfg, "paths.processed_dir") / "manifest.csv"
    if not manifest.exists():
        print(f"{manifest} not found; run `python -m dr.data` first.")
        return 2
    out = resolve_path(cfg, "paths.reports_dir") / "eda"
    for p in save_aggregate_plots(load_manifest(manifest), out):
        print(f"wrote {p}")
    return 0


def save_aggregate_plots(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Save the aggregate (image-free) charts as PNGs for the reports folder."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in AGGREGATE_PLOTS.items():
        fig = fn(df)
        path = out / f"{name}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


if __name__ == "__main__":
    raise SystemExit(main())
