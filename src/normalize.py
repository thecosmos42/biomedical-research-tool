"""Normalize LLM-extracted techniques / biological models via synonym vocabularies.

Usage: python -m src.normalize [--force]
Reads data/interim/extractions_raw.parquet + src/vocab/{technique,model}_synonyms.yaml.
Writes data/interim/entities_llm.parquet (pmid, entity_type, raw, canonical) and data/interim/rare_terms.csv.
Terms that occur fewer than MIN_COUNT times corpus-wide (after mapping) are dropped to rare_terms.csv.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from src.config import ROOT, load_config

MIN_COUNT = 3
VOCAB_DIR = ROOT / "src" / "vocab"
FILES = {"technique": "technique_synonyms.yaml", "biological_model": "model_synonyms.yaml"}


def clean(term: str) -> str:
    """Lowercase, unify dashes/whitespace, strip surrounding punctuation."""
    t = term.lower().replace("\u2013", "-").replace("\u2011", "-")
    t = re.sub(r"\s+", " ", t).strip(" .,;:()[]\"'")
    return t


def load_vocab(path: Path) -> dict[str, str]:
    """synonym -> canonical lookup (canonical maps to itself)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    lookup: dict[str, str] = {}
    for canon, syns in raw.items():
        c = clean(canon)
        lookup[c] = c
        for s in syns or []:
            lookup[clean(s)] = c
    return lookup


def normalize_frame(ext: pd.DataFrame, vocabs: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Explode both list columns into (pmid, entity_type, raw, canonical), de-duplicated per paper."""
    rows = []
    for r in ext.itertuples():
        for etype, col in (("technique", r.techniques), ("biological_model", r.biological_models)):
            seen = set()
            for raw in col:
                canon = vocabs[etype].get(clean(raw), clean(raw))
                if canon and canon not in seen:
                    seen.add(canon)
                    rows.append((r.pmid, etype, raw, canon))
    return pd.DataFrame(rows, columns=["pmid", "entity_type", "raw", "canonical"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = ap.parse_args()

    interim = Path(load_config()["paths"]["interim"])
    out, rare_path = interim / "entities_llm.parquet", interim / "rare_terms.csv"
    if out.exists() and not args.force:
        print(f"{out} exists; skipping (use --force to re-run)")
        return

    ext = pd.read_parquet(interim / "extractions_raw.parquet")
    vocabs = {k: load_vocab(VOCAB_DIR / f) for k, f in FILES.items()}
    df = normalize_frame(ext, vocabs)
    before = df.groupby("entity_type").canonical.nunique().to_dict()

    n_papers = df.groupby(["entity_type", "canonical"]).pmid.nunique().rename("n_papers")
    rare = n_papers[n_papers < MIN_COUNT].reset_index()
    rare.to_csv(rare_path, index=False)
    keep = n_papers[n_papers >= MIN_COUNT].index
    df = df[pd.MultiIndex.from_frame(df[["entity_type", "canonical"]]).isin(keep)].reset_index(drop=True)
    df.to_parquet(out, index=False)

    print(f"rows={len(df)} unique canonical before rare-drop={before} after="
          f"{df.groupby('entity_type').canonical.nunique().to_dict()} rare_dropped={len(rare)}")
    for etype, k in (("technique", 30), ("biological_model", 20)):
        top = df[df.entity_type == etype].groupby("canonical").pmid.nunique().nlargest(k)
        print(f"\ntop {k} {etype}s (papers):\n{top.to_string()}")
    print(f"\nwrote {out}\nwrote {rare_path}")


if __name__ == "__main__":
    main()
