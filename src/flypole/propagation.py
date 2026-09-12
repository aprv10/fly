"""A deliberately simple sparse activity-propagation model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flypole.sparse_graph import SparseConnectome, outgoing_normalized


@dataclass(frozen=True)
class PropagationConfig:
    steps: int = 10
    decay: float = 0.2
    gain: float = 1.0
    stimulus: float = 1.0


def resolve_stimulus_indices(
    graph: SparseConnectome,
    *,
    body_ids: list[int] | None = None,
    indices: list[int] | None = None,
) -> list[int]:
    """Resolve user-provided body IDs and row indices into unique row indices."""
    resolved = list(indices or [])
    id_to_index = {
        int(body_id): index for index, body_id in enumerate(graph.neurons["bodyId"])
    }
    for body_id in body_ids or []:
        try:
            resolved.append(id_to_index[int(body_id)])
        except KeyError as error:
            raise ValueError(f"bodyId {body_id} is not present in this graph") from error

    if not resolved:
        resolved = [0]
    invalid = [index for index in resolved if not 0 <= index < len(graph.neurons)]
    if invalid:
        raise ValueError(f"stimulus indices outside graph bounds: {invalid}")
    return list(dict.fromkeys(resolved))


def simulate(
    graph: SparseConnectome,
    stimulus_indices: list[int],
    *,
    config: PropagationConfig | None = None,
) -> list[np.ndarray]:
    """Apply a pulse at t=0 and return activity snapshots through all timesteps."""
    settings = config or PropagationConfig()
    if settings.steps < 0:
        raise ValueError("steps cannot be negative")
    if not 0 <= settings.decay <= 1:
        raise ValueError("decay must be between 0 and 1")
    if settings.gain < 0 or settings.stimulus < 0:
        raise ValueError("gain and stimulus must be non-negative")

    weights = outgoing_normalized(graph.adjacency)
    activity = np.zeros(len(graph.neurons), dtype=np.float32)
    activity[stimulus_indices] = settings.stimulus
    history = [activity.copy()]

    for _ in range(settings.steps):
        propagated = weights @ activity
        activity = np.tanh(settings.decay * activity + settings.gain * propagated)
        history.append(np.asarray(activity, dtype=np.float32))

    return history


def top_activity(
    graph: SparseConnectome,
    activity: np.ndarray,
    *,
    count: int = 10,
    threshold: float = 1e-6,
) -> list[tuple[int, int, float, str | None, str | None]]:
    """Return the most active neurons with useful metadata for CLI output."""
    candidates = np.flatnonzero(activity > threshold)
    ranked = candidates[np.argsort(activity[candidates])[::-1]][:count]
    rows = []
    for index in ranked:
        neuron = graph.neurons.iloc[int(index)]
        rows.append(
            (
                int(index),
                int(neuron["bodyId"]),
                float(activity[index]),
                _optional_text(neuron.get("type")),
                _optional_text(neuron.get("instance")),
            )
        )
    return rows


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
