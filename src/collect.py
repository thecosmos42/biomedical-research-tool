"""Collect PubMed abstracts via NCBI E-utilities.

Usage: python -m src.collect [--force] [--max N]
Writes data/raw/abstracts.jsonl and data/raw/collection_meta.json.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from Bio import Entrez
from tqdm import tqdm

from src.config import load_config, optional_env, require_env

MIN_ABSTRACT_CHARS = 200


def _text(el: ET.Element | None) -> str:
    """Full text of an element including inline markup children."""
    return "".join(el.itertext()).strip() if el is not None else ""


def _year(article: ET.Element) -> int | None:
    """PubDate year, falling back to MedlineDate then ArticleDate."""
    for path in (".//Journal/JournalIssue/PubDate/Year", ".//Journal/JournalIssue/PubDate/MedlineDate",
                 ".//ArticleDate/Year"):
        m = re.search(r"\d{4}", _text(article.find(path)))
        if m:
            return int(m.group())
    return None


def _abstract(article: ET.Element) -> str:
    parts = []
    for node in article.findall(".//Abstract/AbstractText"):
        text = _text(node)
        label = node.get("Label")
        if text:
            parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts)


def parse_articles(xml_text: str | bytes) -> list[dict]:
    """Parse an efetch PubmedArticleSet XML into records (unfiltered)."""
    root = ET.fromstring(xml_text)
    records = []
    for pa in root.findall(".//PubmedArticle"):
        article = pa.find("MedlineCitation/Article")
        pmid = _text(pa.find("MedlineCitation/PMID"))
        if article is None or not pmid:
            continue
        journal = _text(pa.find("MedlineCitation/MedlineJournalInfo/MedlineTA")) or _text(
            article.find("Journal/Title"))
        mesh = [
            {"descriptor": _text(h.find("DescriptorName")),
             "major": h.find("DescriptorName").get("MajorTopicYN") == "Y"}
            for h in pa.findall("MedlineCitation/MeshHeadingList/MeshHeading")
            if h.find("DescriptorName") is not None
        ]
        doi = next((_text(i) for i in pa.findall(".//ArticleIdList/ArticleId") if i.get("IdType") == "doi"), None)
        records.append({
            "pmid": pmid,
            "title": _text(article.find("ArticleTitle")),
            "abstract": _abstract(article),
            "journal": journal,
            "year": _year(article),
            "doi": doi or None,
            "mesh_terms": mesh,
            "pub_types": [_text(p) for p in article.findall("PublicationTypeList/PublicationType")],
            "languages": [_text(lang) for lang in article.findall("Language")],
        })
    return records


def keep(rec: dict) -> bool:
    """Filter: non-empty title, abstract >= 200 chars, English, year known."""
    return bool(
        rec["title"] and len(rec["abstract"]) >= MIN_ABSTRACT_CHARS and rec["year"] is not None
        and (not rec["languages"] or "eng" in rec["languages"])
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    ap.add_argument("--max", type=int, default=None, help="override pubmed.max_results")
    args = ap.parse_args()

    cfg = load_config()
    raw = Path(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    out, meta_path = raw / "abstracts.jsonl", raw / "collection_meta.json"
    if out.exists() and not args.force:
        print(f"{out} exists; skipping (use --force to re-run)")
        return

    Entrez.email = require_env("ENTREZ_EMAIL")
    api_key = optional_env("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key
    delay = 0.11 if api_key else 0.34

    pm = cfg["pubmed"]
    retmax = args.max or pm["max_results"]
    with Entrez.esearch(db="pubmed", term=pm["query"], retmax=retmax, sort="relevance") as h:
        res = Entrez.read(h)
    ids, count = res["IdList"], int(res["Count"])
    print(f"esearch: {count} matches, fetching {len(ids)}")

    seen: dict[str, dict] = {}
    fetched = 0
    bs = pm["batch_size"]
    for i in tqdm(range(0, len(ids), bs), desc="efetch"):
        batch = ids[i:i + bs]
        for attempt in range(3):
            try:
                with Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="xml") as h:
                    xml = h.read()
                break
            except Exception as e:  # transient network/HTTP errors
                if attempt == 2:
                    raise
                print(f"retry after error: {e}")
                time.sleep(2 * (attempt + 1))
        recs = parse_articles(xml)
        fetched += len(recs)
        for r in recs:
            if keep(r):
                seen.setdefault(r["pmid"], r)
        time.sleep(delay)

    with open(out, "w", encoding="utf-8") as f:
        for r in seen.values():
            r = {k: v for k, v in r.items() if k != "languages"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"query": pm["query"], "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "esearch_count": count, "fetched": fetched, "after_filters": len(seen)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with_mesh = sum(1 for r in seen.values() if r["mesh_terms"])
    print(f"fetched={fetched} kept={len(seen)} with_mesh={with_mesh} "
          f"({with_mesh / max(len(seen), 1):.1%})\nwrote {out}\nwrote {meta_path}")


if __name__ == "__main__":
    main()
