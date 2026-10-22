"""TF-IDF + KMeans baseline.

Usage: python -m src.baseline [--force]
k = number of non-noise HDBSCAN clusters (fairness, see docs/EVALUATION.md).
Writes data/interim/baseline_clusters.parquet and data/interim/baseline_cluster_terms.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import load_config

TOP_TERMS = 10
LABEL_TERMS = 3


def build_tfidf(docs: list[str], params: dict):
    """Fit the TF-IDF vectorizer from config params."""
    p = dict(params)
    p["ngram_range"] = tuple(p["ngram_range"])
    vec = TfidfVectorizer(stop_words="english", lowercase=True, **p)
    return vec, vec.fit_transform(docs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = ap.parse_args()

    cfg = load_config()
    interim = Path(cfg["paths"]["interim"])
    out_c, out_t = interim / "baseline_clusters.parquet", interim / "baseline_cluster_terms.csv"
    if out_c.exists() and not args.force:
        print(f"{out_c} exists; skipping (use --force to re-run)")
        return

    clusters = pd.read_parquet(interim / "clusters.parquet")
    k = clusters.loc[~clusters.is_noise, "cluster_id"].nunique()
    with open(Path(cfg["paths"]["raw"]) / "abstracts.jsonl", encoding="utf-8") as f:
        by_pmid = {r["pmid"]: r for r in (json.loads(line) for line in f if line.strip())}
    pmids = clusters.pmid.tolist()
    docs = [by_pmid[p]["title"] + ". " + by_pmid[p]["abstract"] for p in pmids]

    vec, x = build_tfidf(docs, cfg["baseline"]["tfidf"])
    rs = cfg["evaluation"]["random_state"]
    km = KMeans(n_clusters=k, n_init=10, random_state=rs).fit(x)

    vocab = np.array(vec.get_feature_names_out())
    sizes = np.bincount(km.labels_, minlength=k)
    rows = []
    for c in range(k):
        terms = list(vocab[np.argsort(-km.cluster_centers_[c])[:TOP_TERMS]])
        rows.append({"cluster_id": c, "label": ", ".join(terms[:LABEL_TERMS]), "size": int(sizes[c]),
                     "top_terms": "|".join(terms)})
    pd.DataFrame({"pmid": pmids, "cluster_id": km.labels_.astype(int)}).to_parquet(out_c, index=False)
    pd.DataFrame(rows).to_csv(out_t, index=False)
    print(f"k={k} tfidf_shape={x.shape} sizes min/median/max={sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")
    print(f"wrote {out_c}\nwrote {out_t}")


if __name__ == "__main__":
    main()
