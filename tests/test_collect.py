from pathlib import Path

from src.collect import keep, parse_articles

FIXTURE = Path(__file__).parent / "fixtures" / "efetch_sample.xml"


def test_parse_fields():
    recs = parse_articles(FIXTURE.read_bytes())
    assert [r["pmid"] for r in recs] == ["1001", "1002", "1003"]
    r = recs[0]
    assert r["title"] == "mRNA vaccine immunogenicity in adults."
    assert r["abstract"].startswith("BACKGROUND: mRNA vaccines")
    assert "\nMETHODS: " in r["abstract"]
    assert r["journal"] == "Vaccine J"
    assert r["year"] == 2021
    assert r["doi"] == "10.1000/abc"
    assert r["mesh_terms"] == [{"descriptor": "mRNA Vaccines", "major": True},
                               {"descriptor": "Adult", "major": False}]
    assert r["pub_types"] == ["Journal Article", "Clinical Trial, Phase III"]
    assert recs[1]["year"] == 2019  # MedlineDate fallback
    assert recs[1]["doi"] is None


def test_filters():
    recs = parse_articles(FIXTURE.read_bytes())
    assert [keep(r) for r in recs] == [True, False, False]  # ok, short abstract, non-English
