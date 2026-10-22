"""Entity annotation from PubTator3 precomputed annotations.

Usage: python -m src.annotate [--force] [--limit N]
Reads data/raw/abstracts.jsonl; caches raw BioC JSON in data/raw/pubtator_cache/;
writes data/interim/entities_pubtator.parquet.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.config import load_config

API = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
BATCH = 100
DELAY = 0.4  # PubTator3 allows ~3 requests/s
TYPES = {"disease", "species", "gene", "chemical"}
COLUMNS = ["pmid", "entity_type", "mention", "identifier", "canonical", "n_mentions"]


def parse_biocjson(payload: dict) -> list[dict]:
    """Flatten a PubTator3 export payload into one dict per annotation mention."""
    rows = []
    for doc in payload.get("PubTator3", []):
        pmid = str(doc.get("pmid") or doc.get("id"))
        for passage in doc.get("passages", []):
            for ann in passage.get("annotations", []):
                inf = ann.get("infons", {})
                etype = str(inf.get("type", "")).lower()
                ident = inf.get("identifier")
                if etype not in TYPES or not ident:
                    continue
                rows.append({"pmid": pmid, "entity_type": etype,
                             "identifier": str(ident).removeprefix("MESH:"),
                             "mention": ann.get("text", ""), "name": inf.get("name")})
    return rows


def aggregate(rows: list[dict]) -> pd.DataFrame:
    """Collapse mentions to one row per (pmid, entity_type, identifier)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["pmid"], r["entity_type"], r["identifier"]), []).append(r)
    out = []
    for (pmid, etype, ident), rs in groups.items():
        mention = Counter(r["mention"].lower() for r in rs).most_common(1)[0][0]
        names = [r["name"] for r in rs if r["name"] and str(r["name"]) != ident]
        canonical = Counter(names).most_common(1)[0][0] if names else mention
        out.append((pmid, etype, mention, ident, canonical, len(rs)))
    return pd.DataFrame(out, columns=COLUMNS)


def fetch_batch(pmids: list[str], cache_dir: Path, idx: int) -> dict:
    """Return the BioC payload for a batch, using the on-disk cache."""
    path = cache_dir / f"batch_{idx:04d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            resp = requests.get(API, params={"pmids": ",".join(pmids)}, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            if attempt == 3:
                raise
            print(f"retry batch {idx}: {e}")
            time.sleep(3 * (attempt + 1))
    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(DELAY)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    ap.add_argument("--limit", type=int, default=None, help="annotate only the first N papers")
    args = ap.parse_args()

    cfg = load_config()
    raw, interim = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"])
    out = interim / "entities_pubtator.parquet"
    if out.exists() and not args.force:
        print(f"{out} exists; skipping (use --force to re-run)")
        return

    with open(raw / "abstracts.jsonl", encoding="utf-8") as f:  # not splitlines(): abstracts may contain U+2028
        pmids = [json.loads(line)["pmid"] for line in f if line.strip()]
    if args.limit:
        pmids = pmids[:args.limit]
    cache_dir = raw / "pubtator_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i in tqdm(range(0, len(pmids), BATCH), desc="pubtator"):
        rows += parse_biocjson(fetch_batch(pmids[i:i + BATCH], cache_dir, i // BATCH))
    df = aggregate(rows)
    df.to_parquet(out, index=False)

    covered = df[df.entity_type.isin(["disease", "species"])].pmid.nunique()
    print(f"papers={len(pmids)} rows={len(df)} with_disease_or_species={covered} ({covered / len(pmids):.1%})")
    print("entities per type:", df.entity_type.value_counts().to_dict())
    print("top 20 diseases (papers):")
    print(df[df.entity_type == "disease"].groupby("canonical").pmid.nunique().nlargest(20).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
