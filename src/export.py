"""Export the Power BI star schema (docs/DATA_SCHEMA.md section 3).

Usage: python -m src.export [--force]
Writes fact_publications, dim_topics, dim_entities, bridge_publication_entities, dim_years to data/processed/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config

TABLES = ["fact_publications", "dim_topics", "dim_entities", "bridge_publication_entities", "dim_years"]


def build_entities(pubtator: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """Unified (pmid, entity_type, canonical, identifier) rows from PubTator and LLM extraction."""
    pt = pubtator.copy()
    # one display name per (type, identifier): the most frequent canonical across papers
    names = (pt.groupby(["entity_type", "identifier"]).canonical
             .agg(lambda s: s.value_counts().index[0]).rename("canonical_final").reset_index())
    pt = pt.merge(names, on=["entity_type", "identifier"])
    pt = pt.assign(canonical=pt.canonical_final)[["pmid", "entity_type", "canonical", "identifier"]]
    ll = llm.assign(identifier="")[["pmid", "entity_type", "canonical", "identifier"]]
    return pd.concat([pt, ll], ignore_index=True).drop_duplicates()


def build_star(recs: list[dict], clusters: pd.DataFrame, summary: pd.DataFrame, ext: pd.DataFrame,
               ents: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Assemble the five star-schema tables and validate integrity."""
    base = pd.DataFrame([{"pmid": r["pmid"], "title": r["title"], "journal": r["journal"], "year": int(r["year"]),
                          "doi": r["doi"] or ""} for r in recs])
    fact = (base.merge(clusters[["pmid", "cluster_id", "is_noise", "umap_x", "umap_y"]], on="pmid", how="inner")
            .rename(columns={"cluster_id": "topic_id"})
            .merge(ext[["pmid", "study_type"]], on="pmid", how="left"))
    fact["study_type"] = fact.study_type.fillna("")
    fact["pubmed_url"] = "https://pubmed.ncbi.nlm.nih.gov/" + fact.pmid + "/"

    topics = summary.rename(columns={"cluster_id": "topic_id", "label": "topic_label"})[
        ["topic_id", "topic_label", "top_terms", "size"]]

    dim_e = (ents[["entity_type", "canonical", "identifier"]].drop_duplicates()
             .sort_values(["entity_type", "identifier", "canonical"]).reset_index(drop=True))
    dim_e.insert(0, "entity_id", range(1, len(dim_e) + 1))
    bridge = (ents.merge(dim_e, on=["entity_type", "canonical", "identifier"])[["pmid", "entity_id"]]
              .drop_duplicates())
    bridge = bridge[bridge.pmid.isin(fact.pmid)]
    years = pd.DataFrame({"year": range(int(fact.year.min()), int(fact.year.max()) + 1)})

    out = {"fact_publications": fact, "dim_topics": topics, "dim_entities": dim_e,
           "bridge_publication_entities": bridge, "dim_years": years}
    validate(out)
    return out


def validate(t: dict[str, pd.DataFrame]) -> None:
    """Enforce the integrity rules from DATA_SCHEMA.md; raises AssertionError on violation."""
    f, tp, e, b, y = (t[k] for k in TABLES)
    assert f.pmid.is_unique and tp.topic_id.is_unique and e.entity_id.is_unique and y.year.is_unique
    assert f.topic_id.isin(tp.topic_id).all(), "fact.topic_id missing from dim_topics"
    assert f.year.isin(y.year).all(), "fact.year missing from dim_years"
    assert b.pmid.isin(f.pmid).all() and b.entity_id.isin(e.entity_id).all(), "bridge FK broken"
    assert not b.duplicated().any(), "duplicate bridge pairs"
    for name, df, keys in [("fact", f, ["pmid", "topic_id", "year"]), ("dim_topics", tp, ["topic_id"]),
                           ("dim_entities", e, ["entity_id", "entity_type", "canonical"]),
                           ("bridge", b, ["pmid", "entity_id"])]:
        assert not df[keys].isna().any().any() and not (df[keys].astype(str) == "").any().any(), f"null key in {name}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = ap.parse_args()

    cfg = load_config()
    raw, interim, proc = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"]), Path(cfg["paths"]["processed"])
    proc.mkdir(parents=True, exist_ok=True)
    if (proc / "fact_publications.csv").exists() and not args.force:
        print("star schema exists; skipping (use --force to re-run)")
        return

    with open(raw / "abstracts.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    ents = build_entities(pd.read_parquet(interim / "entities_pubtator.parquet"),
                          pd.read_parquet(interim / "entities_llm.parquet"))
    star = build_star(recs, pd.read_parquet(interim / "clusters.parquet"), pd.read_csv(interim / "cluster_summary.csv"),
                      pd.read_parquet(interim / "extractions_raw.parquet"), ents)
    for name, df in star.items():
        df.to_csv(proc / f"{name}.csv", index=False, encoding="utf-8")
        print(f"{name}: {len(df)} rows -> {proc / (name + '.csv')}")
    e = star["dim_entities"]
    print("entities by type:", e.entity_type.value_counts().to_dict())
    print("integrity checks passed")


if __name__ == "__main__":
    main()
