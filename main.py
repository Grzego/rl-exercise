import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import tyro
from tqdm import tqdm

from agent import BaseAgent, GreedyPolicy, PolicyGradientAgent, Transition
from environment import Environment
from utils import evaluate_agent

matplotlib.use("qtagg")


@dataclass
class Args:
    train_steps: int = 500
    """Number of training steps to run the agent in the environment."""

    transitions_per_step: int = 200
    """Number of transitions to collect in each training step."""

    hidden_size: int = 256
    """Size of the hidden layer in the policy network."""
    learning_rate: float = 0.01
    """Learning rate for the policy gradient updates."""
    entropy_factor: float = 1e-3
    """Factor to scale the entropy penalty for exploration."""
    random_seed: int = 42
    """Random seed for reproducibility."""

    agent_save_path: str = "agent.pkl"
    """Path to save the trained agent."""

    eval_random_seed: int = 1337
    """Random seed for final evaluation."""
    eval_num_episodes: int = 1000
    """Number of episodes to run for final evaluation."""
    output_file: str = "output.txt"
    """File to save the evaluation score."""


def collect_transitions(
    key: jnp.ndarray,
    env: Environment,
    agent: BaseAgent,
    num_steps: int = 2048,
) -> tuple[jnp.ndarray, Transition]:
    """
    Collect transitions from the environment using the agent's policy.
    This function runs the agent in the environment for a specified number of steps,
    collecting observations, actions, rewards, next observations, and done flags.

    :param key: JAX random key for stochastic actions.
    :param env: The environment to interact with.
    :param agent: The agent that will take actions in the environment.
    :param num_steps: The number of steps to run the agent in the environment.

    :return: A tuple containing the updated JAX random key and a batch of transitions.
    """

    transitions: list[Transition] = []

    reset_key, key = jax.random.split(key)
    obs = env.reset(reset_key)

    for _ in range(num_steps):
        action_key, key = jax.random.split(key)
        action = agent.act(action_key, obs)

        next_obs, reward, done = env.step(action)

        transitions.append(
            Transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
        )

        if not done:
            obs = next_obs
        else:
            reset_key, key = jax.random.split(key)
            obs = env.reset(reset_key)

    # stack transitions along the first axis to create a batch, structure remains the same
    return key, Transition(*jax.tree.map(lambda *args: jnp.stack(args, axis=0), *transitions))


def visualize_policy(env: Environment, agent: BaseAgent):
    """
    Visualize the agent's policy by plotting the action vectors on a grid of positions.
    """

    positions = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 2 * jnp.pi, num=200),
            jnp.linspace(0, jnp.pi, num=100),
            indexing="xy",
        ),
        axis=-1,
    )
    actions = agent.act(None, positions, deterministic=True)
    greedy_actions = GreedyPolicy(env).act(None, positions, deterministic=True)

    x_delta = positions[0, 1, 0] - positions[0, 0, 0]
    y_delta = positions[1, 0, 1] - positions[0, 0, 1]
    x_grid = positions[..., 0]
    y_grid = positions[..., 1]

    agent_x_vec = x_delta * env.actions[actions, 0] / 0.3
    agent_y_vec = y_delta * env.actions[actions, 1] / 0.3

    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].quiver(
        x_grid,
        y_grid,
        agent_x_vec,
        agent_y_vec,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=5e-4,
        color="black",
    )
    ax[0].set_xlabel("X Position")
    ax[0].set_ylabel("Y Position")
    ax[0].set_title("Trained Policy")
    ax[0].grid(False)

    greedy_x_vec = x_delta * env.actions[greedy_actions, 0] / 0.3
    greedy_y_vec = y_delta * env.actions[greedy_actions, 1] / 0.3

    ax[1].quiver(
        x_grid,
        y_grid,
        greedy_x_vec,
        greedy_y_vec,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=5e-4,
        color="black",
    )
    ax[1].set_xlabel("X Position")
    ax[1].set_ylabel("Y Position")
    ax[1].set_title("Greedy Policy")
    ax[1].grid(False)

    plt.tight_layout()
    plt.show()


def main():
    args = tyro.cli(Args)
    key = jax.random.PRNGKey(args.random_seed)

    env = Environment()

    agent_key, key = jax.random.split(key)
    agent = PolicyGradientAgent(
        agent_key, env.observation_shape, num_actions=env.num_actions, hidden_size=args.hidden_size
    )

    with tqdm(range(args.train_steps)) as pbar:
        for step in pbar:
            key, transitions = collect_transitions(key, env, agent, num_steps=args.transitions_per_step)
            agent.train(transitions, discount=1.0, learning_rate=args.learning_rate, entropy_factor=args.entropy_factor)

            if step % 10 == 0:
                eval_reward = evaluate_agent(env, agent, num_episodes=20, seed=1337)
                pbar.set_description_str(
                    f"mean reward: {jnp.mean(transitions.reward):6.2f}, eval reward: {eval_reward:6.2f}"
                )

    with open(args.agent_save_path, "wb") as file:
        pickle.dump(agent, file)
    print(f'Trained agent saved to "{args.agent_save_path}"')

    # evaluate the agent's performance
    eval_reward = evaluate_agent(
        env, agent, num_episodes=args.eval_num_episodes, seed=args.eval_random_seed, enable_progressbar=True
    )

    print(f"Average reward over {args.eval_num_episodes} episodes: {eval_reward:.2f}")
    with open(args.output_file, "w") as file:
        file.write(str(eval_reward))

    visualize_policy(env, agent)


if __name__ == "__main__":
    main()
