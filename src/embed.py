"""SPECTER2 embeddings (base model + proximity adapter).

Usage: python -m src.embed [--force] [--limit N]
Writes data/interim/embeddings.npy and data/interim/embedding_index.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config

MAX_LEN = 512


def load_model(model_name: str, adapter: str):
    """Load SPECTER2 base with the proximity adapter active."""
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    model.load_adapter(adapter, source="hf", load_as="proximity", set_active=True)
    return tok, model.eval()


def encode(texts: list[str], tok, model, batch_size: int, device: str) -> np.ndarray:
    """CLS-token embeddings (SPECTER2 convention), float32 [n, 768]."""
    model.to(device)
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="embed"):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            out.append(model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N papers")
    args = ap.parse_args()

    cfg = load_config()
    interim = Path(cfg["paths"]["interim"])
    emb_path, idx_path = interim / "embeddings.npy", interim / "embedding_index.csv"
    if emb_path.exists() and not args.force:
        print(f"{emb_path} exists; skipping (use --force to re-run)")
        return

    with open(Path(cfg["paths"]["raw"]) / "abstracts.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        recs = recs[:args.limit]

    ec = cfg["embedding"]
    tok, model = load_model(ec["model"], ec["adapter"])
    texts = [r["title"] + tok.sep_token + r["abstract"] for r in recs]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"encoding {len(texts)} papers on {device}")
    emb = encode(texts, tok, model, ec["batch_size"], device)

    interim.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    pd.DataFrame({"row": range(len(recs)), "pmid": [r["pmid"] for r in recs]}).to_csv(idx_path, index=False)
    print(f"shape={emb.shape} dtype={emb.dtype} nan={int(np.isnan(emb).sum())}\nwrote {emb_path}\nwrote {idx_path}")


if __name__ == "__main__":
    main()
