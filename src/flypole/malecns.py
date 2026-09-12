"""Bounded access to the public MaleCNS v1.0 neuPrint dataset."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from flypole.sparse_graph import from_dataframes, save_connectome


NEUPRINT_SERVER = "https://neuprint.janelia.org"
MALECNS_DATASET = "male-cns:v1.0"
TOKEN_ENVIRONMENT_VARIABLE = "NEUPRINT_APPLICATION_CREDENTIALS"
MIN_SUBGRAPH_SIZE = 1_000
MAX_SUBGRAPH_SIZE = 5_000


def create_client(*, token: str | None = None) -> Any:
    """Create an authenticated neuPrint client without persisting credentials."""
    from neuprint import Client

    credential = token or os.environ.get(TOKEN_ENVIRONMENT_VARIABLE)
    if not credential:
        raise RuntimeError(
            f"No neuPrint token found. Set {TOKEN_ENVIRONMENT_VARIABLE} in your "
            "environment after copying the token from your neuPrint account page."
        )
    return Client(NEUPRINT_SERVER, dataset=MALECNS_DATASET, token=credential)


def select_high_connectivity_neurons(client: Any, *, size: int) -> pd.DataFrame:
    """Select a deterministic, bounded high-connectivity neuron core."""
    if not MIN_SUBGRAPH_SIZE <= size <= MAX_SUBGRAPH_SIZE:
        raise ValueError(
            f"size must be between {MIN_SUBGRAPH_SIZE} and {MAX_SUBGRAPH_SIZE}"
        )

    query = f"""
    MATCH (n:Neuron)
    WHERE n.bodyId IS NOT NULL AND (n.pre IS NOT NULL OR n.post IS NOT NULL)
    RETURN n.bodyId AS bodyId,
           n.type AS type,
           n.instance AS instance,
           coalesce(n.pre, 0) AS pre,
           coalesce(n.post, 0) AS post
    ORDER BY coalesce(n.pre, 0) + coalesce(n.post, 0) DESC, n.bodyId ASC
    LIMIT {int(size)}
    """
    neurons = client.fetch_custom(query)
    if len(neurons) != size:
        raise RuntimeError(f"requested {size} neurons but neuPrint returned {len(neurons)}")
    if "bodyId" not in neurons.columns:
        raise RuntimeError("neuPrint neuron response is missing bodyId")
    return neurons.reset_index(drop=True)


def _identify_adjacency_table(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    required = {"bodyId_pre", "bodyId_post", "weight"}
    if required.issubset(first.columns):
        return first
    if required.issubset(second.columns):
        return second
    raise RuntimeError(
        "neuPrint adjacency response did not contain bodyId_pre, bodyId_post, and weight"
    )


def fetch_internal_connections(
    client: Any, neuron_ids: list[int], *, min_weight: int = 1
) -> pd.DataFrame:
    """Fetch only aggregate connections within the selected neuron IDs."""
    if min_weight < 1:
        raise ValueError("min_weight must be at least 1")

    from neuprint import fetch_adjacencies

    first, second = fetch_adjacencies(
        neuron_ids,
        neuron_ids,
        min_total_weight=min_weight,
        omit_rois=True,
        weight_props=["weight"],
        batch_size=200,
        client=client,
    )
    return _identify_adjacency_table(first, second).reset_index(drop=True)


def download_subgraph(
    output: str | Path,
    *,
    size: int = 2_000,
    min_weight: int = 1,
    token: str | None = None,
) -> Path:
    """Query and save a bounded MaleCNS high-connectivity induced subgraph."""
    client = create_client(token=token)
    neurons = select_high_connectivity_neurons(client, size=size)
    neuron_ids = neurons["bodyId"].astype("int64").tolist()
    connections = fetch_internal_connections(
        client, neuron_ids, min_weight=min_weight
    )
    graph = from_dataframes(
        neurons,
        connections,
        manifest={
            "dataset": MALECNS_DATASET,
            "server": NEUPRINT_SERVER,
            "selection": "highest pre+post synapse count",
            "requested_neuron_count": size,
            "minimum_connection_weight": min_weight,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "license": "CC-BY",
        },
    )
    return save_connectome(graph, output)

