"""Extract techniques / biological models / study type from abstracts with the Anthropic API.

Usage: python -m src.extract_llm [--limit N] [--force] [--yes]
Run a dry run first with --limit 50. A full run (no --limit) requires --yes after the cost estimate has been reviewed.
Responses are cached in data/interim/llm_cache.sqlite keyed by (pmid, model, PROMPT_VERSION).
Writes data/interim/extractions_raw.parquet (+ extraction_errors.txt).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd
from tqdm import tqdm

from src.config import load_config, require_env

PROMPT_VERSION = "v1"
STUDY_TYPES = {"clinical_trial", "observational", "preclinical_in_vivo", "preclinical_in_vitro",
               "computational", "review", "other"}
SYSTEM = """You extract structured facts from a biomedical abstract. Return ONLY a JSON object, no prose, no code fences:
{"techniques": ["..."], "biological_models": ["..."], "study_type": "..."}

- techniques: laboratory, clinical-design, or computational methods explicitly used or evaluated
  (e.g. "elisa", "neutralization assay", "mrna-lnp platform", "phase 3 randomized controlled trial").
- biological_models: organisms, animals, cell lines or systems the work was done in
  (e.g. "balb/c mice", "human subjects", "vero cells", "organoid").
- study_type: exactly one of clinical_trial | observational | preclinical_in_vivo | preclinical_in_vitro | computational | review | other.
Rules: extract only what is explicitly stated in the abstract; empty lists are fine; lowercase everything;
short noun phrases, no explanations."""

_lock = threading.Lock()


class Refused(Exception):
    """Raised when the model declines to answer. The paper is skipped and left without extractions."""


def open_cache(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS cache(pmid TEXT, model TEXT, prompt_version TEXT, response_json TEXT,
                   input_tokens INT, output_tokens INT, created_at TEXT, PRIMARY KEY (pmid, model, prompt_version))""")
    return con


def cache_get(con: sqlite3.Connection, pmid: str, model: str) -> tuple | None:
    with _lock:
        return con.execute("SELECT response_json, input_tokens, output_tokens FROM cache WHERE pmid=? AND model=? "
                           "AND prompt_version=?", (pmid, model, PROMPT_VERSION)).fetchone()


def cache_put(con: sqlite3.Connection, pmid: str, model: str, resp: dict, tin: int, tout: int) -> None:
    with _lock:
        con.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?)",
                    (pmid, model, PROMPT_VERSION, json.dumps(resp), tin, tout, datetime.now(timezone.utc).isoformat()))
        con.commit()


def parse_response(text: str) -> dict:
    """Parse and validate the model's JSON; raises ValueError if malformed."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object")
    d = json.loads(m.group())
    clean = lambda xs: [str(x).strip().lower() for x in xs if str(x).strip()] if isinstance(xs, list) else None  # noqa: E731
    tech, mods = clean(d.get("techniques")), clean(d.get("biological_models"))
    st = str(d.get("study_type", "")).strip().lower()
    if tech is None or mods is None:
        raise ValueError("lists missing")
    return {"techniques": tech, "biological_models": mods, "study_type": st if st in STUDY_TYPES else "other"}


def call_model(client: anthropic.Anthropic, model: str, max_tokens: int, text: str) -> tuple[dict, int, int]:
    """One extraction with a single nudged retry for malformed JSON. Raises on persistent failure."""
    tin = tout = 0
    nudge = ""
    for attempt in range(2):
        r = client.messages.create(model=model, max_tokens=max_tokens, system=SYSTEM + nudge,
                                   messages=[{"role": "user", "content": text}])
        tin, tout = tin + r.usage.input_tokens, tout + r.usage.output_tokens
        if r.stop_reason == "refusal":
            raise Refused("model refused (safety filter)")
        try:
            return parse_response("".join(b.text for b in r.content if b.type == "text")), tin, tout
        except (ValueError, json.JSONDecodeError):
            nudge = "\n\nYour previous reply was not valid JSON. Return valid JSON only."
    raise ValueError("malformed JSON after retry")


def cost(tin: int, tout: int, prices: dict) -> float:
    return (tin * prices["input"] + tout * prices["output"]) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N papers (dry run)")
    ap.add_argument("--force", action="store_true", help="rewrite parquet even if present (cache still used)")
    ap.add_argument("--yes", action="store_true", help="confirm a full-corpus run after reviewing the estimate")
    args = ap.parse_args()

    cfg = load_config()
    lc = cfg["llm"]
    model, interim = lc["model"], Path(cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)
    out = interim / "extractions_raw.parquet"
    if args.limit is None and out.exists() and not args.force:
        print(f"{out} exists; skipping (use --force to re-run)")
        return

    with open(Path(cfg["paths"]["raw"]) / "abstracts.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        recs = recs[:args.limit]
    con = open_cache(interim / "llm_cache.sqlite")
    fallback = lc.get("fallback_model")
    todo = [r for r in recs if cache_get(con, r["pmid"], model) is None
            and not (fallback and cache_get(con, r["pmid"], fallback))]

    # rough pre-run estimate: ~4 characters per token for the prompt, ~120 output tokens per paper
    avg_in = sum(min(len(r["title"]) + len(r["abstract"]), lc["max_abstract_chars"]) for r in todo) / 4 / max(len(todo), 1) + 250
    est_in, est_out = avg_in * len(todo), 120 * len(todo)
    print(f"model={model} papers={len(recs)} cached={len(recs) - len(todo)} to_call={len(todo)}")
    print(f"ESTIMATE: ~{est_in:,.0f} input + ~{est_out:,.0f} output tokens = ~${cost(est_in, est_out, lc['price_per_mtok']):.2f} "
          f"(prices are config placeholders)")
    if args.limit is None and todo and not args.yes:
        raise SystemExit("Full run needs --yes after a reviewed --limit 50 dry run.")

    client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"), max_retries=6)
    errors: list[str] = []
    tin_sum = tout_sum = 0

    def work(r: dict) -> tuple[str, dict | None, int, int, str | None]:
        text = f"Title: {r['title']}\n\nAbstract: {r['abstract'][:lc['max_abstract_chars']]}"
        used = model
        try:
            try:
                resp, a, b = call_model(client, model, lc["max_tokens"], text)
            except Refused:
                if not fallback:
                    raise
                used = fallback  # only refused papers are retried on the fallback; model recorded per row
                resp, a, b = call_model(client, fallback, lc["max_tokens"], text)
            cache_put(con, r["pmid"], used, resp, a, b)
            return r["pmid"], resp, a, b, None
        except Exception as e:  # logged and skipped; never abort the whole run
            return r["pmid"], None, 0, 0, f"{type(e).__name__}: {str(e)[:150]}"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=lc["concurrency"]) as ex:
        for fut in tqdm(as_completed([ex.submit(work, r) for r in todo]), total=len(todo), desc="extract"):
            pmid, resp, a, b, err = fut.result()
            tin_sum, tout_sum = tin_sum + a, tout_sum + b
            if err:
                errors.append(f"{pmid}\t{err}")

    rows = []
    for r in recs:
        for m in (model, fallback):
            hit = m and cache_get(con, r["pmid"], m)
            if hit:
                rows.append({"pmid": r["pmid"], **json.loads(hit[0]), "extraction_model": m})
                break
    pd.DataFrame(rows, columns=["pmid", "techniques", "biological_models", "study_type", "extraction_model"]).to_parquet(out, index=False)
    (interim / "extraction_errors.txt").write_text("\n".join(errors), encoding="utf-8")

    n = max(len(todo), 1)
    print(f"done in {time.time() - t0:.0f}s: ok={len(todo) - len(errors)} failed={len(errors)} ({len(errors) / n:.1%}) "
          f"saved_rows={len(rows)}")
    print(f"ACTUAL tokens: {tin_sum:,} in / {tout_sum:,} out = ~${cost(tin_sum, tout_sum, lc['price_per_mtok']):.4f} "
          f"(placeholder prices)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()


