import pandas as pd
import pytest

from flypole.malecns import (
    TOKEN_ENVIRONMENT_VARIABLE,
    _identify_adjacency_table,
    create_client,
    select_high_connectivity_neurons,
)


class FakeClient:
    def __init__(self) -> None:
        self.query = ""

    def fetch_custom(self, query: str) -> pd.DataFrame:
        self.query = query
        return pd.DataFrame({"bodyId": range(1_000)})


def test_selection_is_bounded_and_server_limited() -> None:
    client = FakeClient()

    neurons = select_high_connectivity_neurons(client, size=1_000)

    assert len(neurons) == 1_000
    assert "LIMIT 1000" in client.query
    with pytest.raises(ValueError, match="between 1000 and 5000"):
        select_high_connectivity_neurons(client, size=999)


def test_adjacency_table_is_identified_by_columns() -> None:
    neurons = pd.DataFrame({"bodyId": [1, 2]})
    edges = pd.DataFrame(
        {"bodyId_pre": [1], "bodyId_post": [2], "weight": [3]}
    )

    assert _identify_adjacency_table(neurons, edges) is edges
    assert _identify_adjacency_table(edges, neurons) is edges


def test_missing_token_has_actionable_error(monkeypatch) -> None:
    monkeypatch.delenv(TOKEN_ENVIRONMENT_VARIABLE, raising=False)

    with pytest.raises(RuntimeError, match=TOKEN_ENVIRONMENT_VARIABLE):
        create_client()


def test_client_does_not_require_dataset_discovery(monkeypatch) -> None:
    from neuprint import Client

    Client.DATASETS_CACHE.clear()
    client = create_client(token="local-test-token")

    assert client.dataset == "male-cns:v1.0"
    assert "male-cns:v1.0" in Client.DATASETS_CACHE["https://neuprint.janelia.org"]
