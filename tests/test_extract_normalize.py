import json
import pandas as pd

from src.extract_llm import parse_response
from src.normalize import clean, load_vocab, normalize_frame, VOCAB_DIR


def test_parse_response_valid_and_fenced():
    d = parse_response('```json\n{"techniques": ["ELISA "], "biological_models": ["Mice"], "study_type": "Preclinical_in_vivo"}\n```')
    assert d == {"techniques": ["elisa"], "biological_models": ["mice"], "study_type": "preclinical_in_vivo"}


def test_parse_response_bad_study_type_and_malformed():
    assert parse_response(json.dumps({"techniques": [], "biological_models": [], "study_type": "x"}))["study_type"] == "other"
    for bad in ("no json here", '{"techniques": "elisa"}'):
        try:
            parse_response(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_normalize_maps_synonyms_and_dedupes():
    vocabs = {"technique": load_vocab(VOCAB_DIR / "technique_synonyms.yaml"),
              "biological_model": load_vocab(VOCAB_DIR / "model_synonyms.yaml")}
    ext = pd.DataFrame([{"pmid": "1", "techniques": ["Enzyme-Linked Immunosorbent Assay", "ELISA", "novel thing"],
                         "biological_models": ["BALB/c"]}])
    out = normalize_frame(ext, vocabs)
    assert out[out.entity_type == "technique"].canonical.tolist() == ["elisa", "novel thing"]
    assert out[out.entity_type == "biological_model"].canonical.tolist() == ["balb/c mice"]
    assert clean("  (Mice). ") == "mice"
