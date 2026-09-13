"""Command-line entry points for independently verifiable Phase 1 steps."""

from __future__ import annotations

import argparse
from pathlib import Path

from flypole.cartpole import run_cartpole_smoke
from flypole.circuit import download_anatomical_circuit, inspect_circuit_populations
from flypole.demo import propagation_demo_graph
from flypole.malecns import download_subgraph
from flypole.propagation import (
    PropagationConfig,
    resolve_stimulus_indices,
    simulate,
    top_activity,
)
from flypole.sparse_graph import SparseConnectome, load_connectome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flypole", description="FlyPole Phase 1 utilities"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cartpole = subparsers.add_parser(
        "cartpole-smoke", help="run CartPole with random actions"
    )
    cartpole.add_argument("--episodes", type=int, default=1)
    cartpole.add_argument("--max-steps", type=int, default=200)
    cartpole.add_argument("--seed", type=int, default=0)
    cartpole.set_defaults(handler=_cartpole_smoke)

    fetch = subparsers.add_parser(
        "fetch-subgraph", help="download a bounded MaleCNS induced subgraph"
    )
    fetch.add_argument("--size", type=int, default=2_000)
    fetch.add_argument("--min-weight", type=int, default=1)
    fetch.add_argument(
        "--output",
        type=Path,
        help="output directory (default: data/malecns-core-SIZE)",
    )
    fetch.set_defaults(handler=_fetch_subgraph)

    populations = subparsers.add_parser(
        "inspect-populations", help="inspect configured MaleCNS biological populations"
    )
    populations.add_argument(
        "--spec", type=Path, default=Path("configs/malecns-circuit-v1.json")
    )
    populations.set_defaults(handler=_inspect_populations)

    circuit = subparsers.add_parser(
        "fetch-circuit", help="download a bounded sensory-to-descending MaleCNS circuit"
    )
    circuit.add_argument("--size", type=int, default=2_000)
    circuit.add_argument("--min-weight", type=int, default=1)
    circuit.add_argument(
        "--spec", type=Path, default=Path("configs/malecns-circuit-v1.json")
    )
    circuit.add_argument(
        "--output", type=Path, default=Path("data/malecns-circuit-v1-2000")
    )
    circuit.set_defaults(handler=_fetch_circuit)

    info = subparsers.add_parser("graph-info", help="inspect a saved sparse graph")
    info.add_argument("graph", type=Path)
    info.set_defaults(handler=_graph_info)

    demo = subparsers.add_parser(
        "demo-propagation", help="run propagation on a tiny synthetic graph"
    )
    demo.add_argument("--steps", type=int, default=4)
    demo.add_argument("--top", type=int, default=5)
    demo.set_defaults(handler=_demo_propagation)

    propagation = subparsers.add_parser(
        "simulate", help="stimulate neurons and print activity propagation"
    )
    propagation.add_argument("graph", type=Path)
    propagation.add_argument("--body-id", type=int, action="append", default=[])
    propagation.add_argument("--index", type=int, action="append", default=[])
    propagation.add_argument("--steps", type=int, default=10)
    propagation.add_argument("--decay", type=float, default=0.2)
    propagation.add_argument("--gain", type=float, default=1.0)
    propagation.add_argument("--stimulus", type=float, default=1.0)
    propagation.add_argument("--top", type=int, default=10)
    propagation.add_argument("--threshold", type=float, default=1e-6)
    propagation.set_defaults(handler=_simulate)
    from flypole.mvp_cli import add_commands
    add_commands(subparsers)
    return parser


def _cartpole_smoke(args: argparse.Namespace) -> None:
    print("CartPole-v1 smoke test (random actions, no training)")
    for result in run_cartpole_smoke(
        episodes=args.episodes, max_steps=args.max_steps, seed=args.seed
    ):
        observation = ", ".join(f"{value:+.4f}" for value in result.final_observation)
        print(
            f"episode={result.episode} steps={result.steps} "
            f"reward={result.total_reward:.1f} terminated={result.terminated} "
            f"truncated={result.truncated} final_observation=[{observation}]"
        )


def _fetch_subgraph(args: argparse.Namespace) -> None:
    output = args.output or Path(f"data/malecns-core-{args.size}")
    print(
        f"Querying male-cns:v1.0 for {args.size} neurons "
        f"(minimum edge weight {args.min_weight})...",
        flush=True,
    )
    output = download_subgraph(output, size=args.size, min_weight=args.min_weight)
    graph = load_connectome(output)
    print(
        f"saved graph={output} neurons={len(graph.neurons)} "
        f"edges={graph.adjacency.nnz}"
    )


def _inspect_populations(args: argparse.Namespace) -> None:
    print(f"MaleCNS population audit using {args.spec}")
    for row in inspect_circuit_populations(args.spec):
        status = "OK" if row["usable"] else "INSUFFICIENT"
        print(
            f"{row['role']:<20} available={row['available']:<5d} "
            f"selected={row['selected']:<3d} {status}  {row['annotation']}"
        )


def _fetch_circuit(args: argparse.Namespace) -> None:
    print(
        f"Building annotated MaleCNS circuit with {args.size} neurons from {args.spec}...",
        flush=True,
    )
    output = download_anatomical_circuit(
        args.output,
        spec_path=args.spec,
        size=args.size,
        min_weight=args.min_weight,
    )
    graph = load_connectome(output)
    print(
        f"saved graph={output} neurons={len(graph.neurons)} edges={graph.adjacency.nnz} "
        f"inputs={sum(len(ids) for ids in graph.manifest['input_groups'].values())} "
        f"descending_candidates={len(graph.manifest['output_population'])}"
    )


def _graph_info(args: argparse.Namespace) -> None:
    graph = load_connectome(args.graph)
    density = graph.adjacency.nnz / max(1, graph.adjacency.shape[0] ** 2)
    print(f"graph={args.graph}")
    print(f"dataset={graph.manifest.get('dataset', 'unknown')}")
    print(f"neurons={len(graph.neurons)} edges={graph.adjacency.nnz}")
    print(f"shape={graph.adjacency.shape} density={density:.6f}")
    print(f"orientation={graph.manifest.get('matrix_orientation')}")
    print(f"selection={graph.manifest.get('selection', 'unknown')}")
    if graph.manifest.get("input_groups"):
        print(
            "biological_inputs="
            + ", ".join(
                f"{channel}:{len(body_ids)}"
                for channel, body_ids in graph.manifest["input_groups"].items()
            )
        )
        print(
            f"biological_output_candidates={len(graph.manifest.get('output_population', []))}"
        )


def _simulate(args: argparse.Namespace) -> None:
    graph = load_connectome(args.graph)
    indices = resolve_stimulus_indices(
        graph, body_ids=args.body_id, indices=args.index
    )
    _print_simulation(
        graph,
        indices=indices,
        steps=args.steps,
        decay=args.decay,
        gain=args.gain,
        stimulus=args.stimulus,
        top=args.top,
        threshold=args.threshold,
    )


def _demo_propagation(args: argparse.Namespace) -> None:
    graph = propagation_demo_graph()
    print("Synthetic five-neuron demo (not MaleCNS data)")
    _print_simulation(
        graph,
        indices=[0],
        steps=args.steps,
        decay=0.2,
        gain=1.0,
        stimulus=1.0,
        top=args.top,
        threshold=1e-6,
    )


def _print_simulation(
    graph: SparseConnectome,
    *,
    indices: list[int],
    steps: int,
    decay: float,
    gain: float,
    stimulus: float,
    top: int,
    threshold: float,
) -> None:
    print(
        f"graph neurons={len(graph.neurons)} edges={graph.adjacency.nnz}; "
        f"stimulus_indices={indices}"
    )
    history = simulate(
        graph,
        indices,
        config=PropagationConfig(
            steps=steps, decay=decay, gain=gain, stimulus=stimulus
        ),
    )
    for timestep, activity in enumerate(history):
        active = int((activity > threshold).sum())
        print(f"\nt={timestep:02d} active={active} max={activity.max():.6f}")
        for index, body_id, value, neuron_type, instance in top_activity(
            graph, activity, count=top, threshold=threshold
        ):
            label = neuron_type or instance or "untyped"
            print(
                f"  index={index:4d} bodyId={body_id:<12d} "
                f"activity={value:.6f} label={label}"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
