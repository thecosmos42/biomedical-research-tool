"""UMAP -> HDBSCAN clustering of SPECTER2 embeddings.

Usage: python -m src.cluster [--force]
Writes data/interim/clusters.parquet and data/interim/cluster_summary.csv.
Noise points are reassigned to the nearest centroid but flagged with is_noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.feature_extraction.text import CountVectorizer

from src.config import load_config

TOP_TERMS = 10
N_REP = 3
LABEL_TERMS = 3


def centroids(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cluster ids (excluding noise) and their mean vectors."""
    ids = np.array(sorted(set(labels) - {-1}))
    return ids, np.stack([x[labels == c].mean(axis=0) for c in ids])


def assign_noise(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reassign label -1 points to the nearest centroid (Euclidean, reduced space)."""
    ids, cents = centroids(x, labels)
    out = labels.copy()
    noise = labels == -1
    if noise.any():
        d = ((x[noise, None, :] - cents[None, :, :]) ** 2).sum(-1)
        out[noise] = ids[d.argmin(1)]
    return out


def ctfidf_terms(docs: list[str], labels: np.ndarray, ids: np.ndarray, top: int = TOP_TERMS) -> dict[int, list[str]]:
    """Class-based TF-IDF: one concatenated document per cluster."""
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5, lowercase=True)
    counts = cv.fit_transform(docs)
    cls = np.stack([np.asarray(counts[labels == c].sum(axis=0)).ravel() for c in ids]).astype(float)
    tf = cls / cls.sum(axis=1, keepdims=True)
    idf = np.log(1 + cls.sum(axis=1).mean() / np.maximum(cls.sum(axis=0), 1))
    scores = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    return {int(c): list(vocab[np.argsort(-scores[i])[:top]]) for i, c in enumerate(ids)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = ap.parse_args()

    cfg = load_config()
    interim = Path(cfg["paths"]["interim"])
    out_c, out_s = interim / "clusters.parquet", interim / "cluster_summary.csv"
    if out_c.exists() and not args.force:
        print(f"{out_c} exists; skipping (use --force to re-run)")
        return

    emb = np.load(interim / "embeddings.npy")
    pmids = pd.read_csv(interim / "embedding_index.csv", dtype={"pmid": str}).pmid.tolist()
    with open(Path(cfg["paths"]["raw"]) / "abstracts.jsonl", encoding="utf-8") as f:
        by_pmid = {r["pmid"]: r for r in (json.loads(line) for line in f if line.strip())}
    docs = [by_pmid[p]["title"] + ". " + by_pmid[p]["abstract"] for p in pmids]

    cc = cfg["cluster"]
    x8 = umap.UMAP(**cc["umap"]).fit_transform(emb)
    raw_labels = hdbscan.HDBSCAN(**cc["hdbscan"]).fit_predict(x8)
    is_noise = raw_labels == -1
    labels = assign_noise(x8, raw_labels)
    x2 = umap.UMAP(**cc["umap_2d"]).fit_transform(emb)

    ids, _ = centroids(x8, raw_labels)
    # representative papers: closest to the cluster centroid in embedding space (non-noise members only)
    terms = ctfidf_terms(docs, labels, ids)
    rows = []
    for c in ids:
        members = np.where(raw_labels == c)[0]
        cent = emb[members].mean(axis=0)
        near = members[np.argsort(np.linalg.norm(emb[members] - cent, axis=1))[:N_REP]]
        rows.append({"cluster_id": int(c),
                     "label": ", ".join(terms[int(c)][:LABEL_TERMS]),
                     "size": int((labels == c).sum()),
                     "top_terms": "|".join(terms[int(c)]),
                     "rep_pmids": "|".join(pmids[i] for i in near)})
    summary = pd.DataFrame(rows)

    pd.DataFrame({"pmid": pmids, "cluster_id": labels.astype(int), "is_noise": is_noise,
                  "umap_x": x2[:, 0], "umap_y": x2[:, 1]}).to_parquet(out_c, index=False)
    summary.to_csv(out_s, index=False)

    print(f"clusters={len(ids)} noise={is_noise.sum()} ({is_noise.mean():.1%}) papers={len(pmids)}")
    print(summary[["cluster_id", "size", "label"]].to_string(index=False))
    print(f"wrote {out_c}\nwrote {out_s}")


if __name__ == "__main__":
    main()
