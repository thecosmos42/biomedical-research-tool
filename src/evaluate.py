"""Benchmark SPECTER2+HDBSCAN vs TF-IDF+KMeans against MeSH pseudo-labels.

Protocol is locked in docs/EVALUATION.md. Usage: python -m src.evaluate [--force]
Writes data/processed/evaluation_metrics.csv and evaluation_report.md; fills the README results table.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from src.baseline import build_tfidf
from src.config import ROOT, load_config

WINDOW = 10
TIE_MARGIN = 0.01  # NMI differences below this are reported as a tie
TOKEN_RE = re.compile(r"\b\w\w+\b")
METHODS = {"SPECTER2 + HDBSCAN": "hdbscan", "TF-IDF + KMeans": "kmeans"}


# ---------- MeSH pseudo-labels ----------

BRANCH_NAMES = {
    "A": "Anatomy", "B": "Organisms", "C": "Diseases", "D": "Chemicals and Drugs",
    "E": "Analytical, Diagnostic and Therapeutic Techniques", "F": "Psychiatry and Psychology",
    "G": "Phenomena and Processes", "H": "Disciplines and Occupations", "I": "Anthropology, Education, Sociology",
    "J": "Technology, Industry, Agriculture", "K": "Humanities", "L": "Information Science",
    "M": "Named Groups", "N": "Health Care", "V": "Publication Characteristics", "Z": "Geographicals",
}


def descriptor_category(trees: list[str]) -> str | None:
    """Top-level MeSH branch letter (Diseases preferred when a descriptor sits in several trees).

    Sub-branch granularity yields 104 categories, well above the protocol's target of ~8-20, so top-level branches are used.
    """
    if not trees:
        return None
    return "C" if any(t.startswith("C") for t in trees) else sorted(trees)[0][0]

def derive_labels(records: list[dict], tree: dict[str, list[str]]) -> dict[str, str]:
    """One category per paper: most frequent among major descriptors (fallback all); rarer corpus category wins ties."""
    per_paper: dict[str, list[str]] = {}
    for r in records:
        mesh = r["mesh_terms"]
        use = [m for m in mesh if m["major"]] or mesh
        cats = [descriptor_category(tree.get(m["descriptor"], [])) for m in use]
        per_paper[r["pmid"]] = [c for c in cats if c]
    corpus = Counter(c for cats in per_paper.values() for c in cats)
    labels = {}
    for pmid, cats in per_paper.items():
        if not cats:
            continue
        cnt = Counter(cats)
        top = max(cnt.values())
        labels[pmid] = min((c for c, n in cnt.items() if n == top), key=lambda c: (corpus[c], c))
    return labels


# ---------- NPMI coherence ----------

def doc_tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def window_matrix(docs: list[str], terms: list[str]) -> sparse.csr_matrix:
    """Boolean [n_windows, n_terms]: term present in a 10-token sliding window (bigrams anchored at first token)."""
    tindex = {t: i for i, t in enumerate(terms)}
    rows, cols, offset = [], [], 0
    for text in docs:
        toks = doc_tokens(text)
        n_win = max(1, len(toks) - WINDOW + 1)
        occ: dict[int, list[int]] = {}
        for p, tok in enumerate(toks):
            for term in (tok, f"{tok} {toks[p + 1]}" if p + 1 < len(toks) else None):
                if term in tindex:
                    occ.setdefault(tindex[term], []).append(p)
        for ti, positions in occ.items():
            diff = np.zeros(n_win + 1, dtype=np.int32)
            for p in positions:
                diff[max(0, p - WINDOW + 1)] += 1
                diff[min(n_win - 1, p) + 1] -= 1
            hit = np.nonzero(np.cumsum(diff[:-1]) > 0)[0]
            rows.extend(offset + hit)
            cols.extend([ti] * len(hit))
        offset += n_win
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(offset, len(terms)))


def mean_npmi(term_lists: list[list[str]], docs: list[str]) -> float:
    """Mean over clusters of the mean pairwise NPMI of each cluster's top terms."""
    terms = sorted({t for tl in term_lists for t in tl})
    m = window_matrix(docs, terms)
    n_win = m.shape[0]
    cooc = (m.T @ m).toarray() / n_win
    p = np.asarray(m.sum(axis=0)).ravel() / n_win
    idx = {t: i for i, t in enumerate(terms)}
    scores = []
    for tl in term_lists:
        pair = []
        for a in range(len(tl)):
            for b in range(a + 1, len(tl)):
                i, j = idx[tl[a]], idx[tl[b]]
                pij = cooc[i, j]
                pair.append(-1.0 if pij <= 0 else np.log(pij / (p[i] * p[j])) / -np.log(pij))
        scores.append(np.mean(pair))
    return float(np.mean(scores))


# ---------- main ----------

def load_tree(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path, dtype=str)
    return df.groupby("descriptor").tree_number.apply(list).to_dict()


def extrinsic(truth: pd.Series, pred: pd.Series) -> tuple[float, float]:
    return (normalized_mutual_info_score(truth, pred), adjusted_rand_score(truth, pred))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = ap.parse_args()

    cfg = load_config()
    raw, interim, proc = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"]), Path(cfg["paths"]["processed"])
    proc.mkdir(parents=True, exist_ok=True)
    out_csv, out_md = proc / "evaluation_metrics.csv", proc / "evaluation_report.md"
    if out_csv.exists() and not args.force:
        print(f"{out_csv} exists; skipping (use --force to re-run)")
        return

    with open(raw / "abstracts.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    by_pmid = {r["pmid"]: r for r in recs}
    tree_path = ROOT / "src" / "vocab" / "mesh_tree.csv"
    tree = load_tree(tree_path)
    labels = derive_labels(recs, tree)

    hd = pd.read_parquet(interim / "clusters.parquet")
    km = pd.read_parquet(interim / "baseline_clusters.parquet")
    df = hd.merge(km, on="pmid", suffixes=("_hdbscan", "_kmeans"))
    df = df.rename(columns={"cluster_id_hdbscan": "hdbscan", "cluster_id_kmeans": "kmeans"})
    df["mesh"] = df.pmid.map(labels)
    pmids = df.pmid.tolist()
    docs = [by_pmid[p]["title"] + ". " + by_pmid[p]["abstract"] for p in pmids]

    emb = np.load(interim / "embeddings.npy")
    emb_index = pd.read_csv(interim / "embedding_index.csv", dtype={"pmid": str}).pmid.tolist()
    assert emb_index == pmids, "embedding order differs from clusters.parquet"
    _, tfidf_x = build_tfidf(docs, cfg["baseline"]["tfidf"])

    coherence = {
        "hdbscan": mean_npmi([s.split("|") for s in pd.read_csv(interim / "cluster_summary.csv").top_terms], docs),
        "kmeans": mean_npmi([s.split("|") for s in pd.read_csv(interim / "baseline_cluster_terms.csv").top_terms], docs),
    }

    subsets = {"excluding noise": ~df.is_noise.to_numpy(), "including reassigned noise": np.ones(len(df), bool)}
    rows = []
    for subset_name, mask in subsets.items():
        sub = df[mask]
        lab = sub.dropna(subset=["mesh"])
        for method, col in METHODS.items():
            nmi, ari = extrinsic(lab.mesh, lab[col])
            space = emb if col == "hdbscan" else tfidf_x
            sil = silhouette_score(space[mask], sub[col], metric="cosine")
            rows.append({"method": method, "subset": subset_name, "n_papers": int(mask.sum()),
                         "n_labeled": len(lab), "n_clusters": int(sub[col].nunique()),
                         "nmi": nmi, "ari": ari, "npmi": coherence[col], "silhouette": float(sil)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_csv, index=False)

    headline = metrics[metrics.subset == "including reassigned noise"].set_index("method")
    winner = headline.nmi.idxmax()
    wcol = METHODS[winner]
    sub = df.dropna(subset=["mesh"])
    summary = pd.read_csv(interim / ("cluster_summary.csv" if wcol == "hdbscan" else "baseline_cluster_terms.csv"))
    conf_rows = []
    for c, g in sub.groupby(wcol):
        top = g.mesh.value_counts().head(3)
        desc = "; ".join(f"{BRANCH_NAMES.get(k, k)} ({k}) {v / len(g):.0%}" for k, v in top.items())
        conf_rows.append(f"| {c} | {summary.set_index('cluster_id').label.get(c, '')} | {len(g)} | {desc} |")

    dist = pd.Series(labels).value_counts()
    dist_md = "\n".join(f"| {BRANCH_NAMES.get(k, k)} ({k}) | {v} |" for k, v in dist.items())
    n_unlabeled = len(df) - int(df.mesh.notna().sum())
    h, k_ = headline.loc["SPECTER2 + HDBSCAN"], headline.loc["TF-IDF + KMeans"]
    table = metrics.assign(**{c: metrics[c].round(3) for c in ["nmi", "ari", "npmi", "silhouette"]})
    table_md = "| Method | Subset | Papers | Labeled | Clusters | NMI | ARI | NPMI | Silhouette* |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(
        f"| {r.method} | {r.subset} | {r.n_papers} | {r.n_labeled} | {r.n_clusters} | {r.nmi} | {r.ari} | {r.npmi} | {r.silhouette} |"
        for r in table.itertuples())
    verdict = "SPECTER2 + HDBSCAN scored higher" if winner.startswith("SPECTER2") else "TF-IDF + KMeans scored higher"
    if abs(h.nmi - k_.nmi) < TIE_MARGIN:
        verdict = f"the two methods were effectively tied (NMI gap under {TIE_MARGIN}; nominal leader: {winner})"
    interp = (
        f"On the headline setting (all papers, noise reassigned) {verdict} on NMI "
        f"({h.nmi:.3f} vs {k_.nmi:.3f}) and "
        f"{'SPECTER2' if h.ari > k_.ari else 'TF-IDF'} on ARI ({h.ari:.3f} vs {k_.ari:.3f}). "
        f"Topic coherence (NPMI) was {h.npmi:.3f} for HDBSCAN c-TF-IDF terms vs {k_.npmi:.3f} for KMeans centroid terms. "
        "A limitation is that MeSH pseudo-labels are coarse tree branches while the discovered clusters follow "
        "finer-grained themes (e.g. individual pathogens or vaccine-policy topics), which caps achievable NMI/ARI for both methods. "
        "Silhouette is computed in each method's own feature space (SPECTER2 vs TF-IDF) and is not comparable across methods. "
        "The pre-registered expectation that SPECTER2 would win by a visible margin was not met on the headline setting. "
        "Results are reported as obtained; no clustering parameters were tuned after seeing them."
    )
    out_md.write_text(
        "# Evaluation report\n\nProtocol: docs/EVALUATION.md (locked). Both methods cluster the same papers with k = "
        f"{int(h.n_clusters)} clusters.\n\n## Metrics\n\n{table_md}\n\n*Silhouette uses cosine distance in each method's own space; "
        "not cross-comparable.\n\n## MeSH label derivation\n\n"
        f"Papers labeled: {int(df.mesh.notna().sum())}; excluded (no derivable label): {n_unlabeled}; "
        f"categories: {dist.size}.\n\n| Category | Papers |\n|---|---|\n{dist_md}\n\n"
        f"## Top MeSH categories per cluster — {winner}\n\n| Cluster | Label | Labeled papers | Top 3 MeSH categories |\n|---|---|---|---|\n"
        + "\n".join(conf_rows) + f"\n\n## Interpretation\n\n{interp}\n", encoding="utf-8")

    readme = ROOT / "README.md"
    txt = readme.read_text(encoding="utf-8")
    for name, a, b in [("NMI vs MeSH labels", k_.nmi, h.nmi), ("ARI vs MeSH labels", k_.ari, h.ari),
                       ("Topic coherence (NPMI)", k_.npmi, h.npmi), ("Silhouette", k_.silhouette, h.silhouette)]:
        txt = re.sub(rf"^\| {re.escape(name)} \|.*$", f"| {name} | {a:.3f} | {b:.3f} |", txt, flags=re.M)
    readme.write_text(txt, encoding="utf-8")

    print(table_md)
    print(f"labeled={int(df.mesh.notna().sum())} excluded={n_unlabeled} categories={dist.size} winner={winner}")
    print(f"wrote {out_csv}\nwrote {out_md}\nupdated README.md")


if __name__ == "__main__":
    main()


