# Biomedical Research Tool

A Python NLP pipeline that collects vaccine-related biomedical abstracts from PubMed, extracts research topics, techniques, biological models, and disease areas, and groups papers into research themes. The results are exported as a star schema for Power BI.

The corpus covers infectious disease and vaccines, published 2015 to 2025, with roughly 5,000 abstracts. Topic clusters come from SPECTER2 embeddings reduced with UMAP and grouped with HDBSCAN. Each clustering is benchmarked against a TF-IDF and KMeans baseline, using MeSH descriptors as reference labels.

## Why it's useful

Business development teams in biotech and pharma need to notice shifts in a field early, but reading journals by hand is slow and uneven. This tool reads a corpus of abstracts in bulk and turns it into data you can query:

- **Which topics are growing.** Abstracts are grouped into research themes, so a new cluster shows up in the data rather than in someone's reading list.
- **Which techniques and models are in use.** Methods such as ELISA or organoid models are pulled from the text and normalized, so "neutralization assay" and "neutralisation test" count as one technique.
- **Whether the results hold up.** Groupings are compared against a TF-IDF baseline and checked against MeSH labels. Every output row keeps its PubMed ID, so you can trace it back to the source abstract.

The current scope is vaccines and infectious disease, 2015 to 2025, about 5,000 abstracts. The search query lives in `config.yaml`, so pointing the pipeline at another topic is mostly a configuration change. Linking authors' affiliations to companies is planned but not yet implemented.

## Pipeline

```
PubMed E-utilities ──► collect.py ──► data/raw/abstracts.jsonl
                                          │
             ┌────────────────────────────┼─────────────────────────────┐
             ▼                            ▼                             ▼
      annotate.py (PubTator3)      extract_llm.py (Anthropic)     embed.py (SPECTER2)
      diseases, species, genes     techniques, biological models  768-dim vectors
             │                            │                             │
             └──────────┬─────────────────┘                  ┌──────────┴──────────┐
                        ▼                                    ▼                     ▼
                 normalize.py                          cluster.py            baseline.py
                 synonym merging                       UMAP → HDBSCAN        TF-IDF → KMeans
                        │                                    └──────────┬──────────┘
                        │                                               ▼
                        │                                         evaluate.py
                        │                                    NMI, ARI, NPMI, silhouette
                        └───────────────────┬───────────────────────────┘
                                            ▼
                                        export.py
                                   star-schema CSVs → Power BI
```

## Project structure

```
├── config.yaml           # query, paths, and model parameters
├── requirements.txt
├── .env.example          # template for API keys (do not commit .env)
├── src/
│   ├── config.py         # loads config.yaml and .env
│   ├── collect.py        # PubMed retrieval via NCBI E-utilities
│   ├── annotate.py       # PubTator3 entity annotations
│   ├── extract_llm.py    # LLM extraction of techniques and models
│   ├── normalize.py      # synonym merging
│   ├── embed.py          # SPECTER2 embeddings
│   ├── cluster.py        # UMAP + HDBSCAN clustering
│   ├── baseline.py       # TF-IDF + KMeans baseline
│   ├── evaluate.py       # benchmark report
│   ├── export.py         # Power BI star schema
│   └── vocab/            # MeSH tree lookup and synonym vocabularies
├── data/
│   ├── raw/              # API responses (not versioned)
│   ├── interim/          # intermediate artifacts (not versioned)
│   └── processed/        # final tables for Power BI (not versioned)
├── docs/
│   ├── DATA_SCHEMA.md    # file formats and star-schema definition
│   ├── EVALUATION.md     # benchmarking protocol
│   └── DASHBOARD.md      # Power BI dashboard notes
└── tests/                # smoke tests with small fixtures; no network access
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then set ENTREZ_EMAIL and ANTHROPIC_API_KEY (NCBI_API_KEY is optional)
```

## Running the pipeline

Each stage runs as a module and skips work when its output already exists. Pass `--force` to re-run a stage.

```bash
python -m src.collect            # fetch abstracts
python -m src.annotate           # PubTator3 entities
python -m src.embed              # SPECTER2 embeddings
python -m src.cluster            # UMAP + HDBSCAN clusters
python -m src.baseline           # TF-IDF + KMeans baseline
python -m src.evaluate           # benchmark report
python -m src.extract_llm --limit 50   # dry run with cost estimate
python -m src.extract_llm --yes        # full run, after reviewing the estimate
python -m src.normalize          # synonym merging
python -m src.export             # Power BI tables
```

The LLM step is billed per token. Run the dry run first and review the estimate before running the full corpus. Responses are cached in `data/interim/llm_cache.sqlite`, so reruns do not repeat paid calls.

## Testing

```bash
pytest -q
```

## Results

Headline comparison on the full corpus, with noise points reassigned to the nearest cluster:

| Metric | TF-IDF + KMeans | SPECTER2 + HDBSCAN |
|---|---|---|
| NMI vs MeSH labels | 0.128 | 0.127 |
| ARI vs MeSH labels | 0.029 | 0.026 |
| Topic coherence (NPMI) | 0.269 | 0.155 |
| Silhouette | 0.023 | 0.062 |

On this corpus, SPECTER2 with HDBSCAN did not outperform the TF-IDF baseline on the agreement metrics, and its topics were less coherent. Its silhouette score is higher, but silhouette is computed in each method's own feature space, so the two values are not directly comparable. The MeSH reference labels are coarse top-level branches, which limits the agreement scores of both methods. Full details are in `data/processed/evaluation_report.md` after running the benchmark.

## Configuration and secrets

- All tunable parameters live in `config.yaml`.
- API keys are read from `.env` only. Never commit `.env`.
- The NCBI rate limit is 3 requests per second without a key and 10 with one.