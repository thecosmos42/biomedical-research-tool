# Power BI dashboard — build guide

Status: **not built yet** (requires Power BI Desktop; manual step). The CSVs it consumes are produced by
`python -m src.export` into `data/processed/` and validated against `docs/DATA_SCHEMA.md`.

## 1. Import and model

Get Data → Text/CSV (UTF-8) for all five files in `data/processed/`:
`fact_publications`, `dim_topics`, `dim_entities`, `bridge_publication_entities`, `dim_years`.

Relationships (Model view):

| From | To | Cardinality | Direction |
|---|---|---|---|
| `dim_topics[topic_id]` | `fact_publications[topic_id]` | 1:* | single |
| `dim_years[year]` | `fact_publications[year]` | 1:* | single |
| `fact_publications[pmid]` | `bridge_publication_entities[pmid]` | 1:* | single |
| `dim_entities[entity_id]` | `bridge_publication_entities[entity_id]` | 1:* | **both** (so entity slicers filter papers) |

Set `year` as a whole number (not a summarised field); set `pubmed_url` data category to *Web URL*.

**Disease × Technique matrix:** the bridge links papers to one entity at a time, so duplicate `dim_entities` twice
(Power Query → Reference) as `dim_diseases` (filter `entity_type = disease`) and `dim_techniques`
(`entity_type = technique`), and duplicate the bridge likewise (`bridge_diseases`, `bridge_techniques`), each related
to `fact_publications[pmid]`. Rows = disease canonical, columns = technique canonical, values = distinct count of `pmid`.

Suggested measures:

```
Papers = DISTINCTCOUNT(fact_publications[pmid])
Share of Year = DIVIDE([Papers], CALCULATE([Papers], ALL(dim_topics)))
```

## 2. Pages

1. **Literature Map** — scatter: X=`umap_x`, Y=`umap_y`, legend=`topic_label`, tooltip=title/year/journal;
   slicers: `year`, `topic_label`.
2. **Trends** — stacked area: axis=`year`, legend=`topic_label`, values=`[Papers]`.
3. **Techniques** — top-N bar of `dim_entities[canonical]` (filter `entity_type = technique`, Top 15 by `[Papers]`);
   line chart of selected technique(s) by `year`.
4. **Disease × Technique** — matrix as described above; conditional-format values as a heatmap.
5. **Paper Explorer** — table (title, journal, year, `topic_label`, `study_type`, `pubmed_url`); make it a
   drill-through target from the other pages.

## 3. Things to know when reading the visuals

- Topic labels are auto-generated top c-TF-IDF terms; a few clusters are broad/mixed (see PROBLEMS.md).
- 24.8% of papers were HDBSCAN noise reassigned to the nearest centroid; filter `is_noise = FALSE` for core clusters only.
- PubTator maps many "tetanus" mentions to MeSH "Tetany" (≈181 papers) — expect this label in the disease list.
- 38 papers have year 2026 (epub-ahead dates).
- 336 papers' `study_type`/technique/model values came from claude-haiku-4-5 (model refused by haiku-5-5).

## 4. Deliverables still to do (manual)

- Save the `.pbix` at the repo root (or note an OneDrive link here).
- Export a screenshot per page into `docs/img/` and embed them in README.md.
