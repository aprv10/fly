import pandas as pd

from flypole.sparse_graph import from_dataframes, load_connectome, save_connectome


def test_build_save_and_load_sparse_connectome(tmp_path) -> None:
    neurons = pd.DataFrame(
        {
            "bodyId": [10, 11, 12],
            "type": ["source", "middle", "target"],
        }
    )
    connections = pd.DataFrame(
        {
            "bodyId_pre": [10, 11, 999],
            "bodyId_post": [11, 12, 10],
            "weight": [4, 2, 100],
        }
    )

    graph = from_dataframes(neurons, connections, manifest={"dataset": "test"})

    assert graph.adjacency.shape == (3, 3)
    assert graph.adjacency.nnz == 2
    assert graph.adjacency[1, 0] == 4
    assert graph.adjacency[2, 1] == 2

    save_connectome(graph, tmp_path)
    loaded = load_connectome(tmp_path)
    assert loaded.manifest["dataset"] == "test"
    assert (loaded.adjacency != graph.adjacency).nnz == 0

