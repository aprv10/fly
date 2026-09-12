import numpy as np
from flypole.controller import Brain, action
from flypole.learning import load_checkpoint, save_checkpoint, train, watch
from flypole.sparse_graph import save_connectome
from test_controller import graph


def test_checkpoint_roundtrip_and_deterministic_evaluation(tmp_path):
    brain = Brain(None)
    weights = np.array([.1, .2, 1., 1., 0])
    save_checkpoint(tmp_path / "model.npz", weights, brain, {"variant": "baseline"})
    loaded, w, _ = load_checkpoint(tmp_path / "model.npz")
    np.testing.assert_array_equal(w, weights)
    obs = [.1, -.1, .02, .01]
    assert action(w, loaded.observe(obs)) == action(weights, brain.observe(obs))
    first = watch(tmp_path / "model.npz", episodes=2, headless=True)
    assert watch(tmp_path / "model.npz", episodes=2, headless=True) == first


def test_training_updates_only_readout_and_writes_results(tmp_path):
    train(tmp_path, variant="baseline", iterations=1, directions=2)
    _, first, _ = load_checkpoint(tmp_path / "initial.npz")
    _, latest, _ = load_checkpoint(tmp_path / "latest.npz")
    assert np.any(first != latest)
    assert (tmp_path / "evaluation.json").exists()
    assert len((tmp_path / "rewards.csv").read_text().splitlines()) == 2


def test_brain_checkpoint_replays_activity(tmp_path):
    original = graph()
    brain = Brain(original)
    weights = np.arange(brain.size + 1, dtype=float)
    save_connectome(original, tmp_path / "graph")
    save_checkpoint(tmp_path / "model.npz", weights, brain, {"variant": "synthetic"})
    loaded, _, _ = load_checkpoint(tmp_path / "model.npz")
    observation = [.1, -.1, .01, -.03]
    np.testing.assert_allclose(loaded.observe(observation), brain.observe(observation))
    np.testing.assert_allclose(loaded.activity, brain.activity)
