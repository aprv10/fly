import pandas as pd

from flypole.propagation import PropagationConfig, resolve_stimulus_indices, simulate
from flypole.sparse_graph import from_dataframes


def _chain_graph():
    neurons = pd.DataFrame({"bodyId": [10, 11, 12]})
    connections = pd.DataFrame(
        {
            "bodyId_pre": [10, 11],
            "bodyId_post": [11, 12],
            "weight": [2, 1],
        }
    )
    return from_dataframes(neurons, connections)


def test_activity_propagates_along_directed_edges() -> None:
    graph = _chain_graph()
    history = simulate(
        graph,
        [0],
        config=PropagationConfig(steps=2, decay=0.0, gain=1.0),
    )

    assert history[0][0] == 1.0
    assert history[1][1] > 0
    assert history[1][2] == 0
    assert history[2][2] > 0


def test_body_id_resolution() -> None:
    graph = _chain_graph()
    assert resolve_stimulus_indices(graph, body_ids=[12]) == [2]

