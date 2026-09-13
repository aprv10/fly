# FlyPole

FlyPole uses a bounded, anatomy-aware subgraph of the published adult male
*Drosophila* CNS connectome as a fixed neural network for Gymnasium CartPole. It
stimulates annotated sensory populations, propagates activity through real
MaleCNS connections, and trains only a small linear LEFT/RIGHT readout.

This is an experimental model, not a claim that a fruit fly naturally solves
CartPole. The repository explicitly separates published biological data from the
engineered mapping, neural dynamics, action decoder, and learning algorithm.

## What is included

- MaleCNS v1.0 access through the official `neuprint-python` client.
- A reviewable sensory/output specification in
  `configs/malecns-circuit-v1.json`.
- Bounded construction of a 1,000–5,000-neuron sparse circuit; 2,000 is the
  recommended starting size.
- Eight signed CartPole input populations and annotated descending-neuron
  feature candidates.
- Fixed sparse neural propagation and a trainable linear action readout.
- ARS-style reinforcement learning, checkpoints, reward logs, held-out
  evaluation, structural controls, and a live fly-and-neuron viewer.

FlyPole does **not** download the full 166,000-neuron connectivity table, raw EM
imagery, skeletons, meshes, or individual synapse locations.

## Requirements

- Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/).
- Internet access while querying MaleCNS.
- A free account and authentication token from
  [neuPrint](https://neuprint.janelia.org/).
- A desktop session for the live Pygame viewer. Training itself can be headless.

All commands below are PowerShell commands and assume the repository root is the
current directory.

## Quick start: build, train, and watch

### 1. Install the environment

```powershell
cd C:\path\to\fly
uv sync --dev --locked
```

If Windows denies access to the global uv cache, use a repository-local cache:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD ".uv-cache"
uv sync --dev --locked
```

### 2. Configure the neuPrint token

Sign in to [neuPrint](https://neuprint.janelia.org/), open your account/profile
page, and copy the authentication token shown there. For neuPrint, that profile
authentication token is the application credential FlyPole needs.

Create a local environment file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace `paste-your-neuprint-token-here` in `.env` with your token. Do not add
quotes or spaces around it. `.env` is ignored by Git and must never be committed.

Alternatively, set the credential for only the current PowerShell session:

```powershell
$env:NEUPRINT_APPLICATION_CREDENTIALS = "paste-your-neuprint-token-here"
```

When using `.env`, include `--env-file .env` on commands that contact neuPrint.

### 3. Audit the configured biological populations

This makes small read-only annotation queries. It does not download connectivity:

```powershell
uv run --env-file .env flypole inspect-populations
```

Every row should end in `OK`. MaleCNS v1.0 currently gives approximately:

```text
position+            available=3746  selected=16  OK
position-            available=2345  selected=16  OK
velocity+            available=3746  selected=16  OK
velocity-            available=2345  selected=16  OK
angle+               available=247   selected=16  OK
angle-               available=228   selected=16  OK
angular_velocity+    available=73    selected=16  OK
angular_velocity-    available=75    selected=16  OK
descending_neurons   available=1314  selected=256 OK
```

### 4. Build the bounded anatomy-aware graph

```powershell
uv run --env-file .env flypole fetch-circuit --size 2000 --output data/malecns-circuit-v1-2000
```

The request normally takes a few minutes. Progress bars come from batched
neuPrint adjacency queries. Successful output looks like:

```text
saved graph=data\malecns-circuit-v1-2000 neurons=2000 edges=... inputs=128 descending_candidates=256
```

Inspect the saved graph:

```powershell
uv run flypole graph-info data/malecns-circuit-v1-2000
```

You should see `neurons=2000`, eight 16-neuron biological input groups, 256
descending candidates, and this selection label:

```text
selection=bounded annotated sensory-to-descending pathway corridor
```

### 5. Train the readout

Training without live rendering is fastest:

```powershell
uv run flypole train --graph data/malecns-circuit-v1-2000 --output runs/anatomical-v1 --seed 0 --iterations 60 --directions 12
```

During training, FlyPole prints one line per update:

```text
iteration=1/60 steps=... validation=... best=...
```

Training may stop before iteration 60. This is intentional early stopping: after
at least five updates, training ends when the best five-episode validation mean
reaches 495 out of 500. FlyPole then evaluates the best checkpoint on 20 separate
held-out seeds and prints:

```text
held-out mean=.../500 checkpoint=runs\anatomical-v1\best.npz
```

Every `--output` directory must be new or empty. To run another experiment, use a
new name such as `runs/anatomical-seed-1`.

### 6. Watch the trained fly

```powershell
uv run flypole watch runs/anatomical-v1/best.npz --episodes 5
```

The left panel draws a fly avatar at the actual CartPole cart position. The pole
and all motion still use Gymnasium `CartPole-v1` physics. The right panel samples
the sparse neural graph and colors neurons by current activity. Cyan rings denote
input neurons and purple rings denote readout features. The layout is abstract,
not an anatomical reconstruction.

Press Escape or close the window to stop. Evaluation never changes the weights.
For a non-visual evaluation:

```powershell
uv run flypole watch runs/anatomical-v1/best.npz --headless --episodes 20 --seed 400000
```

## What the anatomy-aware graph builder does

The versioned specification maps CartPole channels to MaleCNS annotations:

| Engineered channel | Annotated biological proxy |
| --- | --- |
| Cart position positive/negative | Right/left visual sensory populations |
| Cart velocity positive/negative | Separate right/left visual sensory populations |
| Pole angle positive/negative | Right/left wind-and-gravity mechanosensory populations |
| Pole angular velocity positive/negative | Right/left haltere sensory-ascending populations |
| Readout candidates | Descending neurons from brain to ventral nerve cord |

The builder selects 16 neurons for each of the eight input channels and 256
descending-neuron candidates. It expands up to three weighted connection layers
forward from the inputs and three layers backward from the outputs. Neurons found
from both directions receive priority. Pathway-support neurons fill the remaining
space, followed by high-connectivity neurons only if needed. Finally, it fetches
aggregate internal connections for exactly the requested number of body IDs.

The resulting directory contains:

```text
data/malecns-circuit-v1-2000/
|-- adjacency.npz   # SciPy CSR matrix, W[postsynaptic, presynaptic]
|-- neurons.csv     # body IDs, annotations, and circuit-selection roles
`-- manifest.json   # exact populations, assumptions, query settings, and provenance
```

`manifest.json` stores the resolved input/output body IDs, so checkpoints remain
reproducible even if selection code later changes.

## Neural controller

For observation `[x, x_dot, theta, theta_dot]`, fixed scales
`[2.4, 2.0, 0.21, 2.0]` normalize and clip values to `[-1, 1]`. Each value is split
into nonnegative positive and negative channels, producing eight drives.

The anatomy-aware graph uses six synchronous propagation updates:

```text
state <- tanh(drive + 0.2 * state + normalized_connectivity @ state)
```

Activity resets for every CartPole observation. Connections are normalized by
each presynaptic neuron's outgoing strength. Up to 64 responsive annotated
descending neurons become features. A linear logit selects LEFT when negative and
RIGHT when nonnegative.

Only this linear readout and its bias are trained. MaleCNS connectivity, input
populations, feature identities, and neural dynamics remain fixed.

## Training and saved artifacts

Training uses NumPy augmented-random-search-style two-sided parameter
perturbations rather than PPO or a deep-learning framework. Defaults are 60
updates and 12 perturbation directions: 24 search episodes plus five validation
episodes per update.

Each run directory contains:

```text
runs/anatomical-v1/
|-- initial.npz       # untrained readout
|-- latest.npz        # most recent readout
|-- best.npz          # best validation checkpoint; use this for watching
|-- config.json       # algorithm, seed, and modeling assumptions
|-- rewards.csv       # per-update search and validation results
|-- evaluation.json   # 20 held-out episode rewards
`-- graph/             # exact sparse graph copied for self-contained replay
```

Validation seeds choose `best.npz`; held-out seeds are used only after training.
Reported search step counts include validation rollouts but not viewer or final
held-out rollouts. Optimizer and random-generator state are not saved, so an
interrupted run cannot currently resume in place.

## Compare against controls

Balancing CartPole does not by itself demonstrate that fly wiring helped. Run the
real graph, shuffled connections, a random network, and a direct-observation
baseline across matched seeds:

```powershell
uv run flypole compare --graph data/malecns-circuit-v1-2000 --output runs/comparison --seeds 0 1 2 --iterations 60
```

Results are written to `runs/comparison/summary.csv` and separate run folders.
The shuffled graph preserves each source neuron's outgoing degree and the weight
distribution while permuting targets. The random graph preserves total node,
edge, and weight counts but not individual degrees. The direct baseline has no
connectome and therefore fewer trainable parameters.

## Offline smoke checks

These commands do not require neuPrint:

```powershell
uv run flypole cartpole-smoke --episodes 2 --seed 0
uv run flypole demo-propagation
uv run pytest -q
```

An explicitly synthetic 1,000-neuron graph can exercise the entire training path:

```powershell
uv run flypole make-synthetic --output data/synthetic-1000 --size 1000
uv run flypole train --graph data/synthetic-1000 --variant synthetic --output runs/synthetic
uv run flypole watch runs/synthetic/best.npz
```

Synthetic performance is a software check, not biological evidence.

## Original high-connectivity graph

The earlier Phase 1 selector remains available for comparison:

```powershell
uv run --env-file .env flypole fetch-subgraph --size 2000 --output data/malecns-core-2000
```

It selects neurons only by total pre- plus postsynaptic count. It contains many
central, descending, and motor neurons but generally excludes sensory neurons.
Consequently, its input populations are seeded random neurons rather than the
annotated populations used by `fetch-circuit`.

## Biological data versus model assumptions

Biological MaleCNS data:

- Body IDs and curated neuron annotations.
- Recorded directed neuron-to-neuron connections.
- Aggregate synapse counts.
- Sensory, descending, motor, class, subclass, nerve, and side annotations.

Engineered FlyPole choices:

- Associating CartPole variables with fly sensory modalities.
- Treating positive/negative values as opposing anatomical populations.
- Selecting a bounded pathway corridor and feature count.
- Normalization, six-step `tanh` dynamics, reset behavior, and all-positive edges.
- Mapping the linear readout to LEFT and RIGHT actions.
- ARS training, validation seeds, and early stopping.

MaleCNS includes predicted neurotransmitter fields, but FlyPole does not yet use
them to assign excitatory or inhibitory signs. Anatomical side also does not imply
a literal CartPole direction; the readout learns the action mapping.

## Troubleshooting

### `No neuPrint token found`

Run the neuPrint commands with `--env-file .env`, or set
`NEUPRINT_APPLICATION_CREDENTIALS` in the current shell.

### `/api/dbmeta/datasets` returns HTTP 500

FlyPole uses the fixed published dataset name `male-cns:v1.0` and seeds the
client's dataset cache to bypass that unrelated server-wide discovery endpoint.
Ensure you are running the current repository code with `uv run flypole ...`.

### The graph command appears to pause

Large batched adjacency queries can be quiet between progress updates. Let the
command finish unless it prints a Python traceback. The final success line begins
with `saved graph=`.

### The output directory already exists

Graph files may be safely reused after `graph-info` confirms the correct
selection. For a fresh graph or training run, choose a new output path instead of
overwriting an experiment you want to retain.

### Training stops before `60/60`

That means early stopping reached a validation mean of at least 495/500. The
held-out mean printed immediately afterward is the final evaluation, not an error.

### The viewer does not open

Use a local desktop PowerShell session rather than a headless server. Confirm
Pygame was installed by `uv sync`, or use `watch --headless` for numeric results.

## Current limitations

- The sensory mapping is biologically motivated but remains a CartPole analogy.
- The bounded corridor is a ranking heuristic, not a complete behavioral circuit.
- All aggregate connections are currently treated as positive.
- Neural state resets at every environment step, so there is no temporal memory.
- Only the output readout learns; no plasticity occurs inside the fly network.
- ARS can vary by seed, and matched multi-seed controls are required before
  attributing performance to real MaleCNS topology.
