import json
from pathlib import Path

from src.annotate import aggregate, parse_biocjson

FIXTURE = Path(__file__).parent / "fixtures" / "pubtator_sample.json"


def test_parse_and_aggregate():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8-sig"))
    df = aggregate(parse_biocjson(payload))
    assert set(df.columns) == {"pmid", "entity_type", "mention", "identifier", "canonical", "n_mentions"}
    assert not df.duplicated(["pmid", "entity_type", "identifier"]).any()
    covid = df[(df.pmid == "33127296") & (df.identifier == "D000086382")].iloc[0]
    assert covid.entity_type == "disease"
    assert covid.canonical == "COVID-19"
    assert covid.n_mentions >= 2
    sp = df[(df.pmid == "38754940") & (df.identifier == "10255")].iloc[0]
    assert sp.canonical == "smallpox"  # species name field is a TaxID, so falls back to mention


def test_empty_payload():
    assert parse_biocjson({"PubTator3": []}) == []
