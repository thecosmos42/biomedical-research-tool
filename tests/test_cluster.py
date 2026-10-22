import numpy as np

from src.cluster import assign_noise, ctfidf_terms


def test_assign_noise_nearest_centroid():
    x = np.array([[0.0, 0], [0.1, 0], [10, 10], [10.1, 10], [9, 9], [1, 1]])
    labels = np.array([0, 0, 1, 1, -1, -1])
    out = assign_noise(x, labels)
    assert out.tolist() == [0, 0, 1, 1, 1, 0]


def test_ctfidf_distinguishes_clusters():
    docs = ["malaria parasite mosquito"] * 6 + ["influenza hemagglutinin strain"] * 6
    labels = np.array([0] * 6 + [1] * 6)
    terms = ctfidf_terms(docs, labels, np.array([0, 1]), top=5)
    assert any("malaria" in t for t in terms[0]) and any("influenza" in t for t in terms[1])
    assert not any("malaria" in t for t in terms[1])

