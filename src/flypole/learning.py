"""Small NumPy augmented-random-search trainer for a fixed brain readout."""

import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from flypole.controller import Brain, action, variant_graph
from flypole.sparse_graph import load_connectome, save_connectome


def episode(env, brain, weights, seed, viewer=None, status="evaluation"):
    observation, _ = env.reset(seed=int(seed))
    total = 0.0
    for step in range(500):
        features = brain.observe(observation)
        chosen = action(weights, features)
        if viewer is not None:
            viewer.draw(observation, brain.activity, total, step, status, chosen)
        observation, reward, terminated, truncated, _ = env.step(chosen)
        total += reward
        if terminated or truncated:
            break
    return float(total)


def evaluate(env, brain, weights, seeds):
    return [episode(env, brain, weights, seed) for seed in seeds]


def save_checkpoint(path, weights, brain, metadata):
    # NPZ contains numeric arrays only; load with allow_pickle=False.
    np.savez_compressed(path, weights=weights, groups=brain.groups,
                        features=brain.features, metadata=json.dumps(metadata))


def load_checkpoint(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        graph = None if metadata["variant"] == "baseline" else load_connectome(path.parent / "graph")
        brain = Brain(graph, groups=saved["groups"], features=saved["features"])
        weights = saved["weights"].copy()
    if weights.shape != (brain.size + 1,) or not np.isfinite(weights).all():
        raise ValueError("invalid checkpoint readout")
    return brain, weights, metadata


def train(output, graph_path=None, variant="real", seed=0, iterations=60,
          directions=12, learning_rate=.03, noise=.05, live=False):
    if iterations < 1 or directions < 2 or learning_rate <= 0 or noise <= 0:
        raise ValueError("positive iterations/rates and at least two directions required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"run directory is not empty: {output}; choose a new --output")
    if variant == "baseline":
        brain = Brain(None)
    else:
        if graph_path is None:
            raise ValueError("--graph is required for graph controllers")
        original = load_connectome(graph_path)
        # Same input/output IDs across structural controls for this seed.
        reference = Brain(original, seed=seed)
        graph = variant_graph(original, variant, seed)
        brain = Brain(graph, groups=reference.groups, features=reference.features)
    output.mkdir(parents=True, exist_ok=True)
    if brain.graph is not None:
        save_connectome(brain.graph, output / "graph")
    metadata = dict(version=1, variant=variant, seed=seed, iterations=iterations,
                    directions=directions, learning_rate=learning_rate, noise=noise,
                    algorithm="ARS-style two-sided parameter search",
                    dynamics="two tanh steps, outgoing normalized, per-observation reset",
                    sensory_encoding="eight engineered signed populations; fixed scales",
                    readout="trainable linear binary logit; fixed graph and neuron selection",
                    dataset="none" if brain.graph is None else brain.graph.manifest.get("dataset"),
                    validation_seeds=list(range(100000, 100005)))
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rng = np.random.default_rng(seed)
    weights = np.zeros(brain.size + 1)
    best = weights.copy()
    best_score = -1.0
    env = gym.make("CartPole-v1")
    viewer = None
    total_steps = 0
    total_episodes = 0
    save_checkpoint(output / "initial.npz", weights, brain, metadata)
    try:
        if live:
            from flypole.viewer import Viewer
            viewer = Viewer(brain, fps=0, stride=8)
        with (output / "rewards.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["iteration", "environment_steps", "episodes", "search_mean", "validation_mean", "best_validation_mean"])
            writer.writeheader()
            for iteration in range(iterations):
                deltas = rng.normal(size=(directions, len(weights)))
                returns = np.zeros((directions, 2))
                for i, delta in enumerate(deltas):
                    rollout_seed = int(rng.integers(0, 90000))
                    for j, sign in enumerate((1, -1)):
                        returns[i, j] = episode(env, brain, weights + sign * noise * delta, rollout_seed)
                selected = np.argsort(returns.max(axis=1))[-max(1, directions // 2):]
                std = returns[selected].std()
                if std > 1e-8:
                    weights += learning_rate / (len(selected) * std) * (
                        (returns[selected, 0] - returns[selected, 1]) @ deltas[selected])
                validation = evaluate(env, brain, weights, metadata["validation_seeds"])
                total_steps += int(returns.sum() + sum(validation))
                total_episodes += directions * 2 + len(validation)
                score = float(np.mean(validation))
                snapshot = {**metadata, "iteration": iteration + 1,
                            "validation_mean": score, "environment_steps": total_steps}
                if score > best_score:
                    best_score = score
                    best = weights.copy()
                    save_checkpoint(output / "best.npz", best, brain, snapshot)
                save_checkpoint(output / "latest.npz", weights, brain, snapshot)
                writer.writerow(dict(iteration=iteration + 1, environment_steps=total_steps,
                                     episodes=total_episodes, search_mean=float(returns.mean()),
                                     validation_mean=score, best_validation_mean=best_score))
                stream.flush()
                print(f"iteration={iteration+1}/{iterations} steps={total_steps} validation={score:.1f} best={best_score:.1f}", flush=True)
                if viewer:
                    episode(env, brain, best, 100000, viewer,
                            f"training {iteration+1}/{iterations} | validation best {best_score:.0f}")
                if best_score >= 495 and iteration >= 4:
                    break
        held_out = evaluate(env, brain, best, range(200000, 200020))
        result = {"variant": variant, "seed": seed, "best_validation_mean": best_score,
                  "held_out_rewards": held_out, "held_out_mean": float(np.mean(held_out)),
                  "held_out_seeds": list(range(200000, 200020)),
                  "environment_steps": total_steps, "training_episodes": total_episodes,
                  "held_out_steps": int(sum(held_out))}
        (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"held-out mean={result['held_out_mean']:.1f}/500 checkpoint={output / 'best.npz'}", flush=True)
        return result
    finally:
        env.close()
        if viewer:
            viewer.close()


def watch(checkpoint, episodes=5, seed=300000, headless=False, fps=50):
    if episodes < 1 or fps < 0:
        raise ValueError("episodes must be positive and fps nonnegative")
    brain, weights, metadata = load_checkpoint(checkpoint)
    env = gym.make("CartPole-v1")
    viewer = None
    rewards = []
    try:
        if not headless:
            from flypole.viewer import Viewer
            viewer = Viewer(brain, fps=fps)
        for index in range(episodes):
            reward = episode(env, brain, weights, seed+index, viewer,
                             f"evaluation episode {index+1}/{episodes} | trained iteration {metadata.get('iteration', 0)}")
            rewards.append(reward)
            print(f"episode={index+1} reward={reward:.0f}", flush=True)
    finally:
        env.close()
        if viewer:
            viewer.close()
    print(f"mean_reward={np.mean(rewards):.1f}/500")
    return rewards
