# FlyPole

FlyPole trains a small linear readout of a fixed 1,000–5,000-neuron sparse network
to control Gymnasium CartPole. It includes a live viewer and structural controls.
MaleCNS data must first be fetched with your neuPrint token; synthetic verification
is explicitly labeled and does not establish performance on real MaleCNS data.

## Train and watch the MVP

From this directory, after fetching `data/malecns-core-2000` below:

```powershell
uv sync --dev --locked
uv run flypole train --graph data/malecns-core-2000 --output runs/fly --seed 0 --live
uv run flypole watch runs/fly/best.npz --episodes 5
```

Omit `--live` for faster training. During training the viewer shows the best-policy
rollout after each update, with iteration progress. During evaluation it displays
every step at 50 FPS. Close the window or press Escape to stop; checkpoints already
written remain available. The left view is drawn from the actual Gymnasium state.
The right view samples at most 240 neurons and 350 connections. Layout is abstract,
not anatomical; active means activity greater than 0.0001. Colors scale to the
current frame maximum. Cyan rings mark inputs; purple rings mark readout neurons.

For immediately runnable offline checks:

```powershell
uv run pytest -q
uv run flypole train --variant baseline --output runs/my-baseline
uv run flypole make-synthetic --output data/my-synthetic --size 1000
uv run flypole train --graph data/my-synthetic --variant synthetic --output runs/my-synthetic
uv run flypole watch runs/my-synthetic/best.npz
```

Each output directory must be new or empty. `uv.lock` pins dependencies. If the
global uv cache is inaccessible in a sandbox, set a local cache for the session:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
```

## What learns, and what is modeled

Real data: selected MaleCNS body IDs, neuron annotations, directed connections and
aggregate connection strengths. All weights stay fixed during training.

Engineered assumptions: each CartPole observation is divided by fixed scales
`[2.4, 2.0, 0.21, 2.0]`, clipped to [-1, 1], and split into eight positive/negative
channels. Each channel stimulates eight distinct seeded neurons. Outgoing weights
are normalized within the retained subgraph. Two synchronous tanh updates with a
0.2 carry term propagate activity, resetting state at each observation. All edges
are positive. These choices are not measured fly physiology or natural fly senses.

The readout sees up to 64 downstream neurons, excluding stimulated neurons.
Features are scaled by their fixed unit-pulse responses. No observation bypass is
used for graph policies. Only one binary-action linear logit and its bias learn:
negative selects LEFT, nonnegative selects RIGHT. The direct baseline uses the four
scaled observations instead. Selection, encoding, dynamics, and learning live in
separate modules so internal plasticity can be added later.

Training uses NumPy ARS-style two-sided parameter perturbations, ranked directions,
and reward-standardized updates (not PPO). This reward-driven RL approach avoids a
deep-learning framework for a small linear policy. See
[Mania et al., 2018](https://arxiv.org/abs/1803.07055). Defaults: up to 60 iterations,
12 directions, 24 search episodes plus 5 validation episodes per iteration.
Training ends early after at least 5 updates when best validation mean reaches 495.
All episodes use normal CartPole rewards and the 500-step limit; no expert policy
or shaped reward is supplied. Learning can fluctuate and success is not guaranteed.

Run artifacts:

- `initial.npz`, `latest.npz`, `best.npz`: numeric readout, neuron selections, metadata.
- `graph/`: exact sparse graph used, for self-contained replay (absent for baseline).
- `config.json`: model assumptions, seed, learning settings.
- `rewards.csv`: update, episode/step counts, search and validation means.
- `evaluation.json`: best-policy returns on 20 held-out episode seeds (200000–200019).

Five validation seeds select checkpoints; they are not the held-out test set.
Search step counts include validation but exclude visualization and final test
rollouts. Checkpoints support evaluation; optimizer/RNG-state training resume is
not implemented. Evaluation is deterministic and never updates the readout:

```powershell
uv run flypole watch runs/fly/best.npz --headless --episodes 20 --seed 400000
```

## Comparison experiments

```powershell
uv run flypole compare --graph data/malecns-core-2000 --output runs/comparison --seeds 0 1 2 --iterations 60
```

This trains real, shuffled, random, and direct-baseline controllers and writes
`summary.csv` plus each run's artifacts. Seeds, maximum iteration budgets and
episode-seed protocol match. Early stopping means actual sample counts differ;
compare both reward and environment steps. Structural controls keep the same
input/output neuron IDs and feature count for each seed; response scaling is
recomputed for each graph. Shuffling randomizes targets within each source,
preserving outgoing degree and weights. Random connectivity preserves total
edge count and weight values, not per-neuron degrees. Self-connections are allowed.
The baseline has fewer readout parameters and is intentionally simpler.

Individual controls use `train --variant shuffled`, `--variant random`, or
`--variant baseline`. A high baseline score or similar synthetic performance means
balancing alone is not evidence that fly wiring contributes to the solution.

The next research improvement is anatomically motivated input/output populations
with matched multi-seed controls. Current population assignment is engineered and
the high-connectivity core omits most inputs from the rest of the CNS.

## Setup with uv

```powershell
uv sync --dev
uv run pytest
```

## Verify CartPole

```powershell
uv run flypole cartpole-smoke --episodes 2 --seed 0
```

This uses random actions only and does not render a window.

## Download a bounded MaleCNS subgraph

1. Sign in at <https://neuprint.janelia.org> and copy the token from the Account
   page.
2. Set it for the current PowerShell session (do not commit it):

```powershell
$env:NEUPRINT_APPLICATION_CREDENTIALS = "paste-your-token-here"
```

3. Fetch the default 2,000-neuron high-connectivity core:

```powershell
uv run flypole fetch-subgraph --size 2000 --output data/malecns-core-2000
```

The command queries `male-cns:v1.0` through neuPrint. It first selects the 2,000
neurons with the largest total pre- plus postsynaptic counts, then downloads only
aggregate connections within that selection. It does not download skeletons,
meshes, image volumes, individual synapses, or the complete connection table.
FlyPole uses the fixed published dataset name directly, so a temporary failure of
neuPrint's server-wide dataset-list endpoint does not block this query.

The output directory contains:

- `adjacency.npz`: compressed SciPy CSR matrix, arranged as `W[post, pre]`.
- `neurons.csv`: mapping from matrix row to MaleCNS body ID and annotations.
- `manifest.json`: dataset, selection, size, weight threshold, and matrix metadata.

Inspect the result:

```powershell
uv run flypole graph-info data/malecns-core-2000
```

## Propagate activity

Verify the propagation model immediately on an explicitly synthetic graph:

```powershell
uv run flypole demo-propagation
```

With no selection flags, neuron index 0 receives a one-time pulse:

```powershell
uv run flypole simulate data/malecns-core-2000 --steps 10 --top 10
```

Stimulate one or more explicit graph indices or MaleCNS body IDs:

```powershell
uv run flypole simulate data/malecns-core-2000 --index 0 --index 1 --steps 8
uv run flypole simulate data/malecns-core-2000 --body-id 12781 --steps 8
```

The model normalizes outgoing connection weights, applies a pulse at timestep
zero, and then uses a bounded leaky update. It is a debugging model, not a claim
of biological fidelity. All connections are treated as positive in Phase 1.

## Data attribution

MaleCNS v1.0 was produced by the Janelia FlyEM Project Team and collaborators
and is made available under CC-BY. See <https://male-cns.janelia.org/>.
