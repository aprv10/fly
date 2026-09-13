"""Anatomy-aware, bounded MaleCNS circuit selection.

The published annotations and connectivity are biological data.  The mapping in
the circuit specification, path-depth limits, ranking and CartPole semantics are
explicit modeling choices.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from flypole.controller import CHANNELS
from flypole.malecns import (
    MALECNS_DATASET,
    MAX_SUBGRAPH_SIZE,
    MIN_SUBGRAPH_SIZE,
    NEUPRINT_SERVER,
    create_client,
    fetch_internal_connections,
)
from flypole.sparse_graph import from_dataframes, save_connectome


ANNOTATION_FIELDS = {
    "superclass",
    "class",
    "subclass",
    "rootSide",
    "somaSide",
    "entryNerve",
    "exitNerve",
    "type",
}
NEURON_RETURN = """n.bodyId AS bodyId, n.type AS type, n.instance AS instance,
       n.superclass AS superclass, n.class AS class, n.subclass AS subclass,
       n.rootSide AS rootSide, n.somaSide AS somaSide,
       n.entryNerve AS entryNerve, n.exitNerve AS exitNerve,
       n.consensusNt AS consensusNt, n.predictedNt AS predictedNt,
       coalesce(n.pre, 0) AS pre, coalesce(n.post, 0) AS post"""


def load_circuit_spec(path: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("dataset") != MALECNS_DATASET:
        raise ValueError(f"circuit specification must target {MALECNS_DATASET}")
    groups = spec.get("input_groups", [])
    if [group.get("channel") for group in groups] != CHANNELS:
        raise ValueError(f"input groups must appear in this order: {CHANNELS}")
    if int(spec.get("input_group_size", 0)) < 1:
        raise ValueError("input_group_size must be positive")
    for population in [*groups, spec.get("output_population", {})]:
        criteria = population.get("criteria", {})
        if not criteria or not set(criteria).issubset(ANNOTATION_FIELDS):
            raise ValueError(f"unsupported or empty annotation criteria: {criteria}")
    return spec


def _literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _where(criteria: dict[str, str]) -> str:
    unknown = set(criteria) - ANNOTATION_FIELDS
    if unknown:
        raise ValueError(f"unsupported annotation fields: {sorted(unknown)}")
    return " AND ".join(f"n.{key} = {_literal(str(value))}" for key, value in criteria.items())


def count_population(client: Any, criteria: dict[str, str]) -> int:
    frame = client.fetch_custom(
        f"MATCH (n:Neuron) WHERE {_where(criteria)} RETURN count(n) AS count"
    )
    return int(frame.iloc[0]["count"])


def select_population(
    client: Any,
    criteria: dict[str, str],
    *,
    limit: int,
    offset: int = 0,
) -> pd.DataFrame:
    if limit < 1 or offset < 0:
        raise ValueError("population limit must be positive and offset nonnegative")
    query = f"""
    MATCH (n:Neuron)
    WHERE {_where(criteria)} AND n.bodyId IS NOT NULL
    RETURN {NEURON_RETURN}
    ORDER BY coalesce(n.pre, 0) + coalesce(n.post, 0) DESC, n.bodyId ASC
    SKIP {int(offset)} LIMIT {int(limit)}
    """
    return client.fetch_custom(query).reset_index(drop=True)


def inspect_circuit_populations(
    spec_path: str | Path, *, token: str | None = None
) -> list[dict[str, Any]]:
    spec = load_circuit_spec(spec_path)
    client = create_client(token=token)
    rows: list[dict[str, Any]] = []
    group_size = int(spec["input_group_size"])
    for group in spec["input_groups"]:
        available = count_population(client, group["criteria"])
        requested = group_size + int(group.get("offset", 0))
        rows.append(
            {
                "role": group["channel"],
                "available": available,
                "selected": group_size,
                "usable": available >= requested,
                "annotation": group["biological_proxy"],
            }
        )
    output = spec["output_population"]
    available = count_population(client, output["criteria"])
    rows.append(
        {
            "role": output["name"],
            "available": available,
            "selected": min(available, int(output["limit"])),
            "usable": available >= int(output["limit"]),
            "annotation": output["biological_role"],
        }
    )
    return rows


def _ranked_neighbors(
    client: Any,
    seed_ids: list[int],
    *,
    outgoing: bool,
    limit: int,
) -> pd.DataFrame:
    if not seed_ids:
        return pd.DataFrame(columns=["bodyId", "pathwayWeight"])
    ids = ",".join(str(int(body_id)) for body_id in seed_ids)
    pattern = (
        "(seed:Neuron)-[edge:ConnectsTo]->(n:Neuron)"
        if outgoing
        else "(n:Neuron)-[edge:ConnectsTo]->(seed:Neuron)"
    )
    query = f"""
    MATCH {pattern}
    WHERE seed.bodyId IN [{ids}] AND n.bodyId IS NOT NULL
    RETURN n.bodyId AS bodyId, sum(edge.weight) AS pathwayWeight
    ORDER BY pathwayWeight DESC, n.bodyId ASC
    LIMIT {int(limit)}
    """
    return client.fetch_custom(query).reset_index(drop=True)


def _expand(
    client: Any,
    seeds: list[int],
    *,
    outgoing: bool,
    hops: int,
    width: int,
) -> dict[int, float]:
    scores = {int(body_id): 1.0 for body_id in seeds}
    seen = set(scores)
    frontier = list(seen)
    for depth in range(1, hops + 1):
        neighbors = _ranked_neighbors(
            client, frontier, outgoing=outgoing, limit=width
        )
        next_frontier: list[int] = []
        for row in neighbors.itertuples(index=False):
            body_id = int(row.bodyId)
            score = math.log1p(float(row.pathwayWeight)) / depth
            scores[body_id] = max(scores.get(body_id, 0.0), score)
            if body_id not in seen:
                next_frontier.append(body_id)
                seen.add(body_id)
        frontier = next_frontier
        if not frontier:
            break
    return scores


def _fetch_neurons(client: Any, body_ids: list[int]) -> pd.DataFrame:
    ids = ",".join(str(int(body_id)) for body_id in body_ids)
    query = f"""
    MATCH (n:Neuron)
    WHERE n.bodyId IN [{ids}]
    RETURN {NEURON_RETURN}
    """
    frame = client.fetch_custom(query)
    by_id = frame.set_index(frame["bodyId"].astype("int64"), drop=False)
    missing = [body_id for body_id in body_ids if body_id not in by_id.index]
    if missing:
        raise RuntimeError(f"MaleCNS did not return {len(missing)} selected neurons")
    return by_id.loc[body_ids].reset_index(drop=True)


def _high_connectivity_fill(
    client: Any, *, excluded: set[int], limit: int
) -> list[int]:
    if limit <= 0:
        return []
    exclusions = ",".join(str(body_id) for body_id in sorted(excluded)) or "-1"
    query = f"""
    MATCH (n:Neuron)
    WHERE n.bodyId IS NOT NULL AND NOT n.bodyId IN [{exclusions}]
    RETURN n.bodyId AS bodyId
    ORDER BY coalesce(n.pre, 0) + coalesce(n.post, 0) DESC, n.bodyId ASC
    LIMIT {int(limit)}
    """
    return client.fetch_custom(query)["bodyId"].astype("int64").tolist()


def download_anatomical_circuit(
    output: str | Path,
    *,
    spec_path: str | Path,
    size: int = 2_000,
    min_weight: int = 1,
    token: str | None = None,
) -> Path:
    """Build a bounded sensory-to-descending induced MaleCNS subgraph."""
    if not MIN_SUBGRAPH_SIZE <= size <= MAX_SUBGRAPH_SIZE:
        raise ValueError(
            f"size must be between {MIN_SUBGRAPH_SIZE} and {MAX_SUBGRAPH_SIZE}"
        )
    spec = load_circuit_spec(spec_path)
    client = create_client(token=token)
    group_size = int(spec["input_group_size"])
    input_groups: dict[str, list[int]] = {}
    required: list[int] = []
    for group in spec["input_groups"]:
        frame = select_population(
            client,
            group["criteria"],
            limit=group_size,
            offset=int(group.get("offset", 0)),
        )
        if len(frame) != group_size:
            raise RuntimeError(
                f"{group['channel']} requested {group_size} neurons but found {len(frame)}"
            )
        body_ids = frame["bodyId"].astype("int64").tolist()
        input_groups[group["channel"]] = body_ids
        required.extend(body_ids)

    output_spec = spec["output_population"]
    outputs = select_population(
        client, output_spec["criteria"], limit=int(output_spec["limit"])
    )["bodyId"].astype("int64").tolist()
    if not outputs:
        raise RuntimeError("the configured descending output population is empty")
    required.extend(outputs)
    required = list(dict.fromkeys(required))
    if len(required) > size:
        raise ValueError("input and output populations exceed requested circuit size")

    selection = spec["selection"]
    inputs = [body_id for group in input_groups.values() for body_id in group]
    forward = _expand(
        client,
        inputs,
        outgoing=True,
        hops=int(selection["forward_hops"]),
        width=int(selection["layer_width"]),
    )
    backward = _expand(
        client,
        outputs,
        outgoing=False,
        hops=int(selection["backward_hops"]),
        width=int(selection["layer_width"]),
    )
    required_set = set(required)
    corridor = sorted(
        (set(forward) & set(backward)) - required_set,
        key=lambda body_id: (-(forward[body_id] * backward[body_id]), body_id),
    )
    support = sorted(
        (set(forward) | set(backward)) - required_set - set(corridor),
        key=lambda body_id: (-(forward.get(body_id, 0) + backward.get(body_id, 0)), body_id),
    )
    selected = required + corridor + support
    selected = selected[:size]
    if len(selected) < size:
        selected.extend(
            _high_connectivity_fill(
                client, excluded=set(selected), limit=size - len(selected)
            )
        )
    if len(selected) != size or len(set(selected)) != size:
        raise RuntimeError(f"could not construct {size} unique circuit neurons")

    neurons = _fetch_neurons(client, selected)
    neurons["selectionRole"] = "support"
    neurons.loc[neurons["bodyId"].isin(corridor), "selectionRole"] = "corridor"
    neurons.loc[neurons["bodyId"].isin(outputs), "selectionRole"] = "descending_output"
    for channel, body_ids in input_groups.items():
        neurons.loc[neurons["bodyId"].isin(body_ids), "selectionRole"] = f"input:{channel}"
    connections = fetch_internal_connections(
        client, selected, min_weight=min_weight
    )
    graph = from_dataframes(
        neurons,
        connections,
        manifest={
            "dataset": MALECNS_DATASET,
            "server": NEUPRINT_SERVER,
            "selection": "bounded annotated sensory-to-descending pathway corridor",
            "selection_spec": spec,
            "input_groups": input_groups,
            "output_population_name": output_spec["name"],
            "output_population": outputs,
            "requested_neuron_count": size,
            "minimum_connection_weight": min_weight,
            "corridor_candidate_count": len(corridor),
            "forward_candidate_count": len(forward),
            "backward_candidate_count": len(backward),
            "recommended_propagation_steps": int(
                selection["recommended_propagation_steps"]
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "license": "CC-BY",
            "biological_data": "MaleCNS neuron annotations, directed connectivity, aggregate synapse counts",
            "engineered_assumptions": "CartPole modality mapping, signed laterality, path ranking, dynamics, action decoding, learning",
        },
    )
    return save_connectome(graph, output)
