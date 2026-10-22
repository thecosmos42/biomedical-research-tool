# Evaluation Protocol — locked before clustering

This protocol is decided **before** running clustering (Phase 3) so we are not tuning
the pipeline toward the metric after seeing results. Any deviation must be recorded in
PROGRESS.md with a justification.

## Research question

> Do document embeddings (SPECTER2) + density clustering (HDBSCAN) recover research
> themes that agree with expert curation better than a classical TF-IDF + KMeans baseline?

## Ground truth: MeSH pseudo-labels

PubMed indexers assign MeSH descriptors to every paper; descriptors marked *major topic*
represent what the paper is chiefly about. We derive **one label per paper**:

1. Take the paper's **major-topic** MeSH descriptors (fall back to all descriptors if no
   majors exist).
2. Map each descriptor to a coarse category using the MeSH tree structure: use the
   descriptor's top-level tree branch under category C (Diseases) when available —
   e.g. `C01` infections subtree children like "Virus Diseases" / "Bacterial Infections" —
   otherwise the first letter+2-digit tree prefix (e.g. `D20` for vaccine chemicals).
   Implementation note: fetch tree numbers from the NLM MeSH RDF/API once and cache, or
   use a static descriptor→tree lookup bundled in `src/vocab/mesh_tree.csv`.
3. If a paper maps to multiple categories, take the most frequent category among its
   major descriptors; break ties by corpus-wide category frequency (rarer category wins,
   to preserve minority classes).
4. Papers with no derivable label are **excluded from extrinsic metrics** (report the count).

The label set should end up with roughly 8–20 categories. Report the label distribution.

**Why this is fair:** MeSH labels were never seen by either pipeline's features
(raw text only), and both pipelines are scored against the same labels on the same subset.

## Fairness constraints

1. **Same documents:** both pipelines cluster the identical paper set.
2. **Same k:** KMeans uses k = number of non-noise HDBSCAN clusters. HDBSCAN chooses k
   organically; giving the baseline the same k removes "wrong k" as an excuse.
3. **Same text input:** title + abstract for both.
4. **Fixed seeds:** all random_state values from config; never tune seeds.
5. **Noise handling:** report metrics both (a) excluding HDBSCAN-noise papers and
   (b) including them after nearest-centroid reassignment. The baseline is evaluated on
   the matching subsets.

## Metrics

| Metric | Type | Notes |
|---|---|---|
| **NMI** (normalized mutual information) vs MeSH labels | extrinsic | headline metric; robust to differing cluster counts |
| **ARI** (adjusted Rand index) vs MeSH labels | extrinsic | chance-corrected; complements NMI |
| **Topic coherence (NPMI)** | intrinsic | top-10 terms per cluster (c-TF-IDF for HDBSCAN, centroid top terms for KMeans), NPMI computed over the full corpus with a 10-word sliding window |
| **Silhouette** | intrinsic | each method in its own feature space (SPECTER2 space vs TF-IDF space) — NOT directly comparable across methods; report with this caveat |

## Reporting

`data/processed/evaluation_report.md` must contain:

1. The metrics table (method × metric, with/without noise rows).
2. Label-derivation stats: papers labeled, excluded, category distribution.
3. A confusion-style table: top 3 MeSH categories per discovered cluster for the winner.
4. 3–5 sentences of honest interpretation, including at least one limitation
   (e.g., MeSH granularity does not perfectly match discovered theme granularity;
   silhouette is not cross-comparable).

## Pre-registered expectations (written before results)

- SPECTER2 expected to win NMI/ARI by a visible margin because embeddings capture
  synonymy (e.g., "SARS-CoV-2 immunization" ≈ "COVID-19 vaccine") that TF-IDF cannot.
- If TF-IDF wins or ties, report it honestly and investigate (likely causes: label
  granularity mismatch, too-aggressive UMAP compression, min_cluster_size too large).
  Honest negative results are acceptable; silently re-tuning until the preferred
  method wins is not.
