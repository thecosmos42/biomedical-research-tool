from src.evaluate import derive_labels, descriptor_category, mean_npmi


def test_descriptor_category_prefers_diseases():
    assert descriptor_category(["D12.776", "C01.925"]) == "C"
    assert descriptor_category(["E02.095"]) == "E"
    assert descriptor_category([]) is None


def test_derive_labels_majors_and_tiebreak():
    tree = {"Flu": ["C01.925"], "Vaccines": ["D20.1"], "Adult": ["M01.060"]}
    recs = [
        {"pmid": "1", "mesh_terms": [{"descriptor": "Flu", "major": True}, {"descriptor": "Adult", "major": False}]},
        {"pmid": "2", "mesh_terms": [{"descriptor": "Adult", "major": False}]},  # fallback to all
        {"pmid": "3", "mesh_terms": [{"descriptor": "Unknown", "major": True}]},  # no label
        {"pmid": "4", "mesh_terms": [{"descriptor": "Flu", "major": True}, {"descriptor": "Vaccines", "major": True}]},
    ]
    lab = derive_labels(recs, tree)
    assert lab["1"] == "C" and lab["2"] == "M" and "3" not in lab
    assert lab["4"] in {"C", "D"}


def test_npmi_higher_for_cooccurring_terms():
    docs = ["malaria parasite mosquito bed nets"] * 5 + ["influenza strain hemagglutinin"] * 5
    good = mean_npmi([["malaria", "parasite"]], docs)
    bad = mean_npmi([["malaria", "influenza"]], docs)
    assert good > bad
