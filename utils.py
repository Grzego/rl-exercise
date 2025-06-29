import jax
from tqdm import tqdm

from agent import BaseAgent
from environment import Environment


def evaluate_agent(
    env: Environment, agent: BaseAgent, num_episodes: int = 10, seed: int = 42, enable_progressbar: bool = False
) -> float:
    """
    Evaluate the agent's performance in the environment over a specified number of episodes.

    :param env: The environment in which to evaluate the agent.
    :param agent: The agent to evaluate.
    :param num_episodes: The number of episodes to run for evaluation.
    :param seed: Random seed for reproducibility.

    :return: The average reward obtained by the agent over the specified number of episodes.
    """
    total_reward = 0.0
    rng_keys = jax.random.split(jax.random.PRNGKey(seed), num=num_episodes)

    for key in tqdm(rng_keys, disable=not enable_progressbar, desc="evaluating agent"):
        obs = env.reset(key)
        done = False
        while not done:
            action = agent.act(None, obs, deterministic=True)
            obs, reward, done = env.step(action)
            total_reward += reward

    return total_reward / num_episodes
