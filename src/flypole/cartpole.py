"""Small, training-free CartPole checks."""

from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    final_observation: tuple[float, ...]


def run_cartpole_smoke(
    *, episodes: int = 1, max_steps: int = 200, seed: int = 0
) -> list[EpisodeResult]:
    """Run CartPole with random actions to verify the environment works."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    env = gym.make("CartPole-v1")
    results: list[EpisodeResult] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            env.action_space.seed(seed + episode)
            total_reward = 0.0
            terminated = False
            truncated = False
            steps = 0

            for steps in range(1, max_steps + 1):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break

            results.append(
                EpisodeResult(
                    episode=episode,
                    steps=steps,
                    total_reward=total_reward,
                    terminated=terminated,
                    truncated=truncated,
                    final_observation=tuple(float(value) for value in observation),
                )
            )
    finally:
        env.close()

    return results

