"""Reference profile: what "normal" inputs and outputs look like, for drift monitoring (Phase 7).

Built from a random sample of *training* images (``reference.n_images``):

* image statistics (the same ``dr.preprocess`` function the served model applies to uploads),
* a PCA of the network's ``embedding`` layer (``reference.pca_components``, normally 32), whose
  fitted object is saved and also packaged inside the served model, so production embeddings
  are projected into exactly the same space,
* the score distribution and the positive-prediction rate at the operating threshold,
* the share of poor-quality images.

Logged to MLflow as the ``reference_profile/`` artifact directory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dr import preprocess as P

PROFILE_TABLE = "reference.parquet"
PCA_FILE = "pca.joblib"
SUMMARY_FILE = "summary.json"


def embed_paths(model, png_paths: list[str], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """(embeddings, scores) for cached PNGs, loading one batch at a time to bound memory."""
    from dr.models import predict_embeddings

    embs, scores = [], []
    for i in range(0, len(png_paths), batch_size):
        batch = np.stack([P.load_png(p) for p in png_paths[i : i + batch_size]])
        e, s = predict_embeddings(model, batch, batch_size=batch_size)
        embs.append(e)
        scores.append(s)
    return np.concatenate(embs), np.concatenate(scores)


def emb_columns(k: int) -> list[str]:
    return [f"emb_{i}" for i in range(k)]


def build_reference(
    model, train_df: pd.DataFrame, cfg: Mapping[str, Any], threshold: float, out_dir: Path
) -> tuple[PCA, dict[str, Any]]:
    """Fit the embedding PCA and write the reference profile into ``out_dir``."""
    rcfg = cfg.get("reference", {})
    seed = int(cfg.get("project", {}).get("seed", 42))
    n = min(int(rcfg.get("n_images", 1000)), len(train_df))
    sample = train_df.sample(n=n, random_state=seed).reset_index(drop=True)

    emb, scores = embed_paths(model, sample["png"].tolist())
    k = int(min(int(rcfg.get("pca_components", 32)), emb.shape[1], len(sample)))
    pca = PCA(n_components=k, random_state=seed).fit(emb)
    proj = pca.transform(emb)

    table = sample[["image", "label", "quality", *P.STAT_COLUMNS]].copy()
    table["p_referable"] = scores
    table["prediction"] = (scores >= threshold).astype(int)
    table = pd.concat([table, pd.DataFrame(proj, columns=emb_columns(k))], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / PROFILE_TABLE, index=False)
    joblib.dump(pca, out_dir / PCA_FILE)
    summary = {
        "n_images": int(n),
        "source": "random sample of the training split",
        "threshold": float(threshold),
        "positive_rate": float(table["prediction"].mean()),
        "label_rate": float(table["label"].mean()),
        "poor_quality_share": float((table["quality"] != P.QUALITY_OK).mean()),
        "score_quantiles": {str(q): float(np.quantile(scores, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "image_stats": {
            c: {"mean": float(table[c].mean()), "std": float(table[c].std())} for c in P.STAT_COLUMNS
        },
        "pca_components": k,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "embedding_dim": int(emb.shape[1]),
    }
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pca, summary
