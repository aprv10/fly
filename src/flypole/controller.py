"""Engineered sensory populations and fixed sparse brain features."""

import numpy as np
from scipy import sparse

from flypole.sparse_graph import SparseConnectome, outgoing_normalized

SCALES = np.array([2.4, 2.0, 0.21, 2.0], dtype=np.float32)
CHANNELS = [f"{name}{sign}" for name in ("position", "velocity", "angle", "angular_velocity") for sign in ("+", "-")]


def encode(observation):
    values = np.clip(np.asarray(observation, dtype=np.float32) / SCALES, -1, 1)
    return np.stack((np.maximum(values, 0), np.maximum(-values, 0)), axis=1).ravel()


class Brain:
    """Two synchronous tanh steps per observation; reset between observations.

    Only downstream neurons are read. There is no raw observation bypass.
    Population assignment, normalization and dynamics are modeling choices.
    """

    def __init__(self, graph: SparseConnectome | None, seed=0, groups=None, features=None):
        self.graph = graph
        self.activity = np.zeros(0 if graph is None else len(graph.neurons), dtype=np.float32)
        if graph is None:
            self.groups = np.empty((8, 0), dtype=int)
            self.features = np.arange(4)
            self.size = 4
            return
        n = len(graph.neurons)
        if not 1000 <= n <= 5000:
            raise ValueError("controller graphs must contain 1000–5000 neurons")
        self.weights = outgoing_normalized(graph.adjacency)
        rng = np.random.default_rng(seed)
        if groups is None:
            eligible = np.flatnonzero(np.asarray(graph.adjacency.sum(axis=0)).ravel() > 0)
            if len(eligible) < 64:
                raise ValueError("graph needs at least 64 neurons with outgoing connections")
            groups = rng.choice(eligible, 64, replace=False).reshape(8, 8)
        self.groups = np.asarray(groups, dtype=int)
        probes = np.stack([self._propagate(row) for row in np.eye(8)])
        strength = probes.max(axis=0)
        strength[self.groups.ravel()] = 0
        if features is None:
            features = np.argsort(-strength, kind="stable")[:64]
            features = features[strength[features] > 1e-9]
        self.features = np.asarray(features, dtype=int)
        if not len(self.features):
            raise ValueError("no downstream features reachable from the input populations")
        self.scale = np.maximum(probes[:, self.features].max(axis=0), 1e-6)
        self.size = len(self.features)
        self.activity.fill(0)

    def _propagate(self, channels):
        drive = np.zeros_like(self.activity)
        drive[self.groups] = np.asarray(channels)[:, None]
        state = np.tanh(drive)
        for _ in range(2):
            state = np.tanh(drive + 0.2 * state + self.weights @ state)
        self.activity = state
        return state.copy()

    def observe(self, observation):
        if self.graph is None:
            return np.clip(np.asarray(observation) / SCALES, -1, 1)
        return self._propagate(encode(observation))[self.features] / self.scale


def action(weights, features):
    """One linear logit suffices for two actions: LEFT=0, RIGHT=1."""
    return int(np.dot(weights[:-1], features) + weights[-1] >= 0)


def variant_graph(graph, variant, seed=0):
    """Controls preserve node IDs, node count, edge count and weight distribution.

    Shuffled permutes targets independently per source (out-degree preserved).
    Random assigns edges uniformly without replacement (degrees not preserved).
    """
    if variant == "real":
        if graph.manifest.get("dataset") != "male-cns:v1.0":
            raise ValueError("real mode requires a male-cns:v1.0 graph")
        return graph
    if variant == "synthetic":
        if graph.manifest.get("dataset") != "synthetic":
            raise ValueError("synthetic mode requires synthetic data")
        return graph
    rng = np.random.default_rng(seed)
    matrix = graph.adjacency.tocsc()
    n = matrix.shape[0]
    if variant == "shuffled":
        rows = np.concatenate([rng.choice(n, matrix.indptr[i+1]-matrix.indptr[i], replace=False) for i in range(n)])
        columns = np.repeat(np.arange(n), np.diff(matrix.indptr))
    elif variant == "random":
        pairs = rng.choice(n*n, matrix.nnz, replace=False)
        rows, columns = pairs // n, pairs % n
    else:
        raise ValueError(f"unknown graph variant: {variant}")
    adjacency = sparse.coo_matrix((matrix.data.copy(), (rows, columns)), shape=(n, n)).tocsr()
    return SparseConnectome(graph.neurons.copy(), adjacency, {**graph.manifest, "control_variant": variant, "control_seed": seed})
