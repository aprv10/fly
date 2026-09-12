"""MVP training, evaluation and controlled experiment commands."""

from pathlib import Path
import csv
import numpy as np
import pandas as pd
from scipy import sparse

from flypole.learning import train, watch
from flypole.sparse_graph import SparseConnectome, save_connectome


def add_commands(subparsers):
    training = subparsers.add_parser("train", help="train only the small linear readout")
    training.add_argument("--graph", type=Path, default=Path("data/malecns-core-2000"))
    training.add_argument("--variant", choices=["real", "shuffled", "random", "baseline", "synthetic"], default="real")
    training.add_argument("--output", type=Path, default=Path("runs/fly"))
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--iterations", type=int, default=60)
    training.add_argument("--directions", type=int, default=12)
    training.add_argument("--live", action="store_true", help="show best-policy rollout after each update")
    training.set_defaults(handler=lambda a: train(a.output, a.graph, a.variant, a.seed, a.iterations, a.directions, live=a.live))
    watching = subparsers.add_parser("watch", help="evaluate a checkpoint without training")
    watching.add_argument("checkpoint", type=Path)
    watching.add_argument("--episodes", type=int, default=5)
    watching.add_argument("--seed", type=int, default=300000)
    watching.add_argument("--fps", type=int, default=50)
    watching.add_argument("--headless", action="store_true")
    watching.set_defaults(handler=lambda a: watch(a.checkpoint, a.episodes, a.seed, a.headless, a.fps))
    synthetic = subparsers.add_parser("make-synthetic", help="create explicitly synthetic test connectivity")
    synthetic.add_argument("--output", type=Path, default=Path("data/synthetic-1000"))
    synthetic.add_argument("--size", type=int, default=1000)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.set_defaults(handler=_synthetic)
    comparison = subparsers.add_parser("compare", help="run all four controls with matched seeds")
    comparison.add_argument("--graph", type=Path, default=Path("data/malecns-core-2000"))
    comparison.add_argument("--output", type=Path, default=Path("runs/comparison"))
    comparison.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    comparison.add_argument("--iterations", type=int, default=60)
    comparison.set_defaults(handler=_compare)


def _synthetic(args):
    if not 1000 <= args.size <= 5000:
        raise ValueError("size must be 1000–5000")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be empty")
    rng = np.random.default_rng(args.seed)
    n = args.size
    pairs = rng.choice(n*n, n*12, replace=False)
    w = sparse.coo_matrix((rng.integers(1, 10, len(pairs)).astype(np.float32),
                           (pairs // n, pairs % n)), shape=(n, n)).tocsr()
    save_connectome(SparseConnectome(pd.DataFrame({"bodyId": np.arange(n)}), w,
                    {"dataset": "synthetic", "seed": args.seed,
                     "matrix_orientation": "rows=postsynaptic, columns=presynaptic"}), args.output)
    print(f"saved SYNTHETIC graph: {args.output}")


def _compare(args):
    results = []
    for seed in args.seeds:
        for variant in ("real", "shuffled", "random", "baseline"):
            result = train(args.output / f"{variant}-{seed}", args.graph, variant, seed, args.iterations)
            results.append({k: result[k] for k in ("variant", "seed", "best_validation_mean", "held_out_mean", "environment_steps")})
            with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
