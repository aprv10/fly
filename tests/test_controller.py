import numpy as np
import pandas as pd
from scipy import sparse
from flypole.controller import Brain, encode, variant_graph
from flypole.sparse_graph import SparseConnectome


def graph():
    rng = np.random.default_rng(1)
    w = sparse.random(1000, 1000, density=.01, random_state=rng, format="csr")
    return SparseConnectome(pd.DataFrame({"bodyId": np.arange(1000)}), w, {"dataset": "synthetic"})


def test_signed_encoding():
    np.testing.assert_allclose(encode([2.4, -2, .21, -2]), [1, 0, 0, 1, 1, 0, 0, 1])


def test_downstream_features_and_fixed_graph():
    g = graph()
    before = g.adjacency.copy()
    brain = Brain(g)
    assert not set(brain.features) & set(brain.groups.ravel())
    features = brain.observe([.1, .2, -.1, -.2])
    assert np.isfinite(features).all() and np.any(features)
    assert (before != g.adjacency).nnz == 0
    np.testing.assert_array_equal(brain.observe(np.zeros(4)), np.zeros(brain.size))


def test_controls_preserve_size_weights_and_outdegree():
    g = graph()
    for mode in ("shuffled", "random"):
        control = variant_graph(g, mode)
        assert control.adjacency.nnz == g.adjacency.nnz
        np.testing.assert_array_equal(np.sort(control.adjacency.data), np.sort(g.adjacency.data))
    np.testing.assert_array_equal(variant_graph(g, "shuffled").adjacency.getnnz(axis=0), g.adjacency.getnnz(axis=0))
