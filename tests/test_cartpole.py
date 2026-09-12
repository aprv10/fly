from flypole.cartpole import run_cartpole_smoke


def test_cartpole_smoke_runs_without_rendering() -> None:
    [result] = run_cartpole_smoke(episodes=1, max_steps=5, seed=7)

    assert 1 <= result.steps <= 5
    assert result.total_reward == result.steps
    assert len(result.final_observation) == 4

