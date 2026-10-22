# Data Schema — file formats and Power BI star schema

Every file the pipeline reads or writes is specified here. If a module needs a new
field, update this document in the same commit.

---

## 1. Raw layer (`data/raw/`)

### `abstracts.jsonl` — one JSON object per line (written by collect.py)

| Field | Type | Notes |
|---|---|---|
| `pmid` | string | primary key, unique |
| `title` | string | required, non-empty |
| `abstract` | string | required, ≥200 chars; structured sections joined as `LABEL: text` |
| `journal` | string | ISO abbreviation if available, else full title |
| `year` | int | publication year; PubDate year, fallback ArticleDate |
| `doi` | string \| null | |
| `mesh_terms` | list of objects | `[{"descriptor": "mRNA Vaccines", "major": true}, ...]` |
| `pub_types` | list of strings | e.g. `["Journal Article", "Clinical Trial, Phase I"]` |

### `collection_meta.json`

`{"query": "...", "retrieved_at": "ISO datetime", "esearch_count": int, "fetched": int, "after_filters": int}`

### `pubtator_cache/` — raw BioC JSON responses, one file per PMID batch (cache only, not consumed downstream)

---

## 2. Interim layer (`data/interim/`)

### `entities_pubtator.parquet` (annotate.py)

| Column | Type | Notes |
|---|---|---|
| `pmid` | string | FK → abstracts |
| `entity_type` | string | `disease` \| `species` \| `gene` \| `chemical` (lowercased from PubTator types) |
| `mention` | string | most frequent surface form in that paper |
| `identifier` | string | normalized ID (MeSH for disease/chemical, NCBI TaxID for species, NCBI Gene ID) |
| `canonical` | string | preferred name resolved from identifier when available, else mention |
| `n_mentions` | int | mention count within the paper |

One row per (pmid, entity_type, identifier).

### `embeddings.npy` + `embedding_index.csv` (embed.py)

- `embeddings.npy`: float32 array, shape `[n_docs, 768]`
- `embedding_index.csv`: columns `row` (int), `pmid` (string) — row i of the array is paper pmid[i]

### `clusters.parquet` (cluster.py)

| Column | Type | Notes |
|---|---|---|
| `pmid` | string | |
| `cluster_id` | int | final assignment (noise reassigned to nearest centroid) |
| `is_noise` | bool | True if HDBSCAN originally labeled -1 |
| `umap_x`, `umap_y` | float | 2D visualization coordinates (separate 2D UMAP) |

### `cluster_summary.csv` (cluster.py)

`cluster_id, label, size, top_terms (pipe-separated), rep_pmids (pipe-separated, 3 PMIDs)`

### `baseline_clusters.parquet` (baseline.py)

`pmid (string), cluster_id (int)` — plus `baseline_cluster_terms.csv` mirroring cluster_summary.

### `llm_cache.sqlite` (extract_llm.py)

Table `cache(pmid TEXT, model TEXT, prompt_version TEXT, response_json TEXT, input_tokens INT, output_tokens INT, created_at TEXT, PRIMARY KEY (pmid, model, prompt_version))`

### `extractions_raw.parquet` (extract_llm.py)

`pmid, techniques (list<string>), biological_models (list<string>), study_type (string)`

### `entities_llm.parquet` (normalize.py)

| Column | Type | Notes |
|---|---|---|
| `pmid` | string | |
| `entity_type` | string | `technique` \| `biological_model` |
| `raw` | string | as extracted by the LLM |
| `canonical` | string | after synonym mapping; lowercased |

---

## 3. Processed layer (`data/processed/`) — the Power BI star schema

```
                    ┌──────────────┐
                    │  dim_topics  │
                    └──────┬───────┘
                           │ topic_id (1:*)
┌────────────┐      ┌──────┴──────────────┐      ┌─────────────────────────────┐
│ dim_years  ├──────┤  fact_publications  ├──────┤ bridge_publication_entities │
└────────────┘ year └─────────────────────┘ pmid └──────────────┬──────────────┘
                                                                │ entity_id (*:1)
                                                         ┌──────┴───────┐
                                                         │ dim_entities │
                                                         └──────────────┘
```

### `fact_publications.csv` — one row per paper

| Column | Type | Notes |
|---|---|---|
| `pmid` | string | PK |
| `title` | string | |
| `journal` | string | |
| `year` | int | FK → dim_years |
| `doi` | string | may be empty |
| `topic_id` | int | FK → dim_topics (SPECTER2 pipeline assignment) |
| `is_noise` | bool | |
| `umap_x`, `umap_y` | float | for the scatter visual |
| `study_type` | string | from LLM extraction; empty if unavailable |
| `pubmed_url` | string | `https://pubmed.ncbi.nlm.nih.gov/{pmid}/` |

### `dim_topics.csv`

| Column | Type |
|---|---|
| `topic_id` | int (PK) |
| `topic_label` | string |
| `top_terms` | string (pipe-separated) |
| `size` | int |

### `dim_entities.csv` — one row per unique canonical entity

| Column | Type | Notes |
|---|---|---|
| `entity_id` | int (PK) | surrogate key |
| `entity_type` | string | `disease` \| `species` \| `gene` \| `chemical` \| `technique` \| `biological_model` |
| `canonical` | string | display name |
| `identifier` | string | MeSH/TaxID where applicable, else empty |

### `bridge_publication_entities.csv`

`pmid (FK), entity_id (FK)` — unique pairs. This bridge enables the Disease × Technique
matrix: filter entities to `disease` on rows and `technique` on columns via two role-playing
references to dim_entities (or two filtered dimension tables if simpler in Power BI).

### `dim_years.csv`

`year (int, PK)` — continuous range min(year)..max(year) so trend axes have no gaps.

### Integrity rules (export.py must enforce before writing)

1. Every `fact_publications.topic_id` exists in `dim_topics`.
2. Every bridge `pmid` exists in fact, every `entity_id` in dim_entities.
3. No null PKs; `pmid` unique in fact; (pmid, entity_id) unique in bridge.
4. Row counts printed for every table.
