"""Conversion and storage for neuron-level sparse connectivity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


REQUIRED_NEURON_COLUMNS = {"bodyId"}
REQUIRED_CONNECTION_COLUMNS = {"bodyId_pre", "bodyId_post", "weight"}


@dataclass(frozen=True)
class SparseConnectome:
    """A neuron table and W[post, pre] sparse weight matrix."""

    neurons: pd.DataFrame
    adjacency: sparse.csr_matrix
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.neurons)
        if self.adjacency.shape != (count, count):
            raise ValueError(
                f"adjacency shape {self.adjacency.shape} does not match {count} neurons"
            )
        if not REQUIRED_NEURON_COLUMNS.issubset(self.neurons.columns):
            raise ValueError("neuron table must contain a bodyId column")
        if self.neurons["bodyId"].duplicated().any():
            raise ValueError("neuron bodyIds must be unique")


def from_dataframes(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    *,
    manifest: dict[str, Any] | None = None,
) -> SparseConnectome:
    """Build a CSR matrix from neuPrint-style neuron and connection tables."""
    missing_neurons = REQUIRED_NEURON_COLUMNS - set(neurons.columns)
    missing_connections = REQUIRED_CONNECTION_COLUMNS - set(connections.columns)
    if missing_neurons:
        raise ValueError(f"missing neuron columns: {sorted(missing_neurons)}")
    if missing_connections:
        raise ValueError(f"missing connection columns: {sorted(missing_connections)}")

    nodes = neurons.copy().reset_index(drop=True)
    nodes["bodyId"] = nodes["bodyId"].astype("int64")
    if nodes["bodyId"].duplicated().any():
        raise ValueError("neuron bodyIds must be unique")

    id_to_index = pd.Series(nodes.index, index=nodes["bodyId"])
    edges = connections.loc[
        connections["bodyId_pre"].isin(id_to_index.index)
        & connections["bodyId_post"].isin(id_to_index.index),
        ["bodyId_pre", "bodyId_post", "weight"],
    ].copy()
    edges = edges[edges["weight"] > 0]
    edges = edges.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()

    rows = edges["bodyId_post"].map(id_to_index).to_numpy(dtype=np.int64)
    columns = edges["bodyId_pre"].map(id_to_index).to_numpy(dtype=np.int64)
    weights = edges["weight"].to_numpy(dtype=np.float32)
    adjacency = sparse.coo_matrix(
        (weights, (rows, columns)), shape=(len(nodes), len(nodes)), dtype=np.float32
    ).tocsr()
    adjacency.sum_duplicates()

    graph_manifest = dict(manifest or {})
    graph_manifest.update(
        {
            "neuron_count": len(nodes),
            "edge_count": int(adjacency.nnz),
            "matrix_orientation": "rows=postsynaptic, columns=presynaptic",
            "weight_dtype": "float32",
        }
    )
    return SparseConnectome(nodes, adjacency, graph_manifest)


def save_connectome(graph: SparseConnectome, directory: str | Path) -> Path:
    """Save a connectome as CSR NPZ, neuron CSV, and JSON manifest."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "adjacency.npz", graph.adjacency, compressed=True)
    graph.neurons.to_csv(output / "neurons.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(graph.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def load_connectome(directory: str | Path) -> SparseConnectome:
    """Load and validate a saved sparse connectome."""
    source = Path(directory)
    adjacency = sparse.load_npz(source / "adjacency.npz").tocsr()
    neurons = pd.read_csv(source / "neurons.csv")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    return SparseConnectome(neurons, adjacency, manifest)


def outgoing_normalized(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Normalize W[post, pre] so each non-empty presynaptic column sums to one."""
    outgoing_strength = np.asarray(adjacency.sum(axis=0)).ravel()
    inverse = np.zeros_like(outgoing_strength, dtype=np.float32)
    nonzero = outgoing_strength > 0
    inverse[nonzero] = 1.0 / outgoing_strength[nonzero]
    return (adjacency @ sparse.diags(inverse, format="csr")).tocsr()

