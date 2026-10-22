import pandas as pd
import pytest

from src.export import build_entities, build_star, validate


def _inputs():
    recs = [{"pmid": "1", "title": "a", "journal": "J", "year": 2020, "doi": None},
            {"pmid": "2", "title": "b", "journal": "J", "year": 2022, "doi": "10/x"}]
    clusters = pd.DataFrame({"pmid": ["1", "2"], "cluster_id": [0, 1], "is_noise": [False, True],
                             "umap_x": [0.1, 0.2], "umap_y": [0.3, 0.4]})
    summary = pd.DataFrame({"cluster_id": [0, 1], "label": ["x", "y"], "size": [1, 1], "top_terms": ["a|b", "c|d"],
                            "rep_pmids": ["1", "2"]})
    ext = pd.DataFrame({"pmid": ["1"], "study_type": ["review"]})
    pt = pd.DataFrame({"pmid": ["1", "2"], "entity_type": ["disease", "disease"], "identifier": ["D1", "D1"],
                       "canonical": ["Flu", "flu"], "mention": ["", ""], "n_mentions": [1, 1]})
    llm = pd.DataFrame({"pmid": ["1"], "entity_type": ["technique"], "canonical": ["elisa"], "raw": ["ELISA"]})
    return recs, clusters, summary, ext, build_entities(pt, llm)


def test_star_schema_builds_and_validates():
    star = build_star(*_inputs())
    f = star["fact_publications"]
    assert f.set_index("pmid").study_type.to_dict() == {"1": "review", "2": ""}
    assert star["dim_years"].year.tolist() == [2020, 2021, 2022]
    assert len(star["dim_entities"]) == 2  # D1 unified across papers + elisa
    assert len(star["bridge_publication_entities"]) == 3


def test_validate_catches_broken_fk():
    star = build_star(*_inputs())
    star["fact_publications"].loc[0, "topic_id"] = 99
    with pytest.raises(AssertionError):
        validate(star)
