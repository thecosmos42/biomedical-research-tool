"""One-off: download MeSH descriptor -> tree number lookup from NLM SPARQL into src/vocab/mesh_tree.csv.

The endpoint truncates results at 1,000 rows and OFFSET paging returns duplicates, so we query by
tree-number prefix and subdivide any prefix that hits the cap. Requests are retried because the
endpoint occasionally drops connections.

Usage: python -m src.vocab.build_mesh_tree
"""
from __future__ import annotations

import csv
import io
import time
from pathlib import Path

import requests

SPARQL = "https://id.nlm.nih.gov/mesh/sparql"
CAP = 1000
OUT = Path(__file__).parent / "mesh_tree.csv"
QUERY = """
PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?l ?tn WHERE {{ ?d a meshv:TopicalDescriptor ; rdfs:label ?l ; meshv:treeNumber ?t . ?t rdfs:label ?tn
FILTER(STRSTARTS(?tn, "{p}")) }}
"""
NEXT_CHARS = "0123456789."
ROOTS = "ABCDEFGHIJKLMNVZ"  # MeSH descriptor tree categories


def fetch(prefix: str) -> list[tuple[str, str]]:
    for attempt in range(6):
        try:
            r = requests.get(SPARQL, params={"query": QUERY.format(p=prefix), "format": "CSV"}, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 5:
                raise
            print(f"retry {prefix}: {type(e).__name__}", flush=True)
            time.sleep(3 * (attempt + 1))
    time.sleep(0.3)
    return [(a, b) for a, b in list(csv.reader(io.StringIO(r.text)))[1:]]


def collect(prefix: str) -> set[tuple[str, str]]:
    rows = fetch(prefix)
    if len(rows) < CAP:
        return set(rows)
    out: set[tuple[str, str]] = set(r for r in rows if r[1] == prefix)
    for ch in NEXT_CHARS:
        out |= collect(prefix + ch)
    return out


def main() -> None:
    rows: set[tuple[str, str]] = set()
    for root in ROOTS:
        got = collect(root)
        print(f"{root}: {len(got)} rows", flush=True)
        rows |= got
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["descriptor", "tree_number"])
        w.writerows(sorted(rows))
    print(f"wrote {OUT} ({len(rows)} rows, {len({d for d, _ in rows})} descriptors)")


if __name__ == "__main__":
    main()
