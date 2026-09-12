"""Small synthetic fixtures for offline demonstrations."""

import pandas as pd

from flypole.sparse_graph import SparseConnectome, from_dataframes


def propagation_demo_graph() -> SparseConnectome:
    """Return a five-neuron branching graph; this is not MaleCNS data."""
    neurons = pd.DataFrame(
        {
            "bodyId": [101, 102, 103, 104, 105],
            "type": ["stimulus", "branch-a", "branch-b", "merge", "output"],
            "instance": [None] * 5,
        }
    )
    connections = pd.DataFrame(
        {
            "bodyId_pre": [101, 101, 102, 103, 104],
            "bodyId_post": [102, 103, 104, 104, 105],
            "weight": [3, 1, 2, 4, 5],
        }
    )
    return from_dataframes(
        neurons,
        connections,
        manifest={"dataset": "synthetic-demo", "selection": "five-neuron branch"},
    )
