import pickle
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import optax
import tyro
from einops import rearrange
from tqdm import tqdm

from agent import BaseAgent, GreedyPolicy, Params, PolicyGradientAgent, Transition
from environment import Environment
from utils import evaluate_agent

matplotlib.use("qtagg")
jax.config.update("jax_enable_x64", True)


@dataclass
class Args:
    train_steps: int = 10000
    """Number of training steps to run the agent in the environment."""

    num_envs: int = 50
    """Number of parallel environments to collect transitions from."""

    transitions_per_step: int = 10
    """Number of transitions to collect in each training step per environment."""

    hidden_size: int = 256
    """Size of the hidden layer in the policy network."""
    learning_rate: float = 2e-3
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


@partial(jax.jit, static_argnames=("env", "agent", "num_envs", "num_steps"))
def collect_transitions(
    key: jnp.ndarray,
    env: Environment,
    agent: BaseAgent,
    agent_params: Params,
    num_envs: int = 1,
    num_steps: int = 100,
) -> tuple[jnp.ndarray, Transition]:
    """
    Collect transitions from the environment using the agent's policy.
    This function runs the agent in the environment for a specified number of steps,
    collecting observations, actions, rewards, next observations, and done flags.

    :param key: JAX random key for stochastic actions.
    :param env: The environment to interact with.
    :param agent: The agent that will take actions in the environment.
    :param agent_params: Parameters of the agent's policy.
    :param num_envs: The number of parallel environments to collect transitions from.
    :param num_steps: The number of steps to run the agent in the environment.

    :return: A tuple containing the updated JAX random key and a batch of transitions.
    """

    transitions: list[Transition] = []

    reset_key, key = jax.random.split(key)
    state, obs = env.reset(reset_key, num=num_envs)

    for _ in range(num_steps):
        action_key, key = jax.random.split(key)
        action = agent.act(action_key, agent_params, obs)

        state, next_obs, reward, done = env.step(state, action)

        transitions.append(
            Transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
        )

        reset_key, key = jax.random.split(key)
        state, obs = env.masked_reset(reset_key, state, mask=done)

    # stack transitions along the first axis to create a batch, structure remains the same
    return key, Transition(
        *jax.tree.map(lambda *args: rearrange(jnp.stack(args, axis=-2), "b s ... -> (b s) ..."), *transitions)
    )


def visualize_policy(env: Environment, agent: BaseAgent, agent_params: Params):
    """
    Visualize the agent's policy by plotting the action vectors on a grid of positions.
    """

    positions = jnp.stack(
        jnp.meshgrid(
            jnp.arange(0, 2 * jnp.pi, step=0.05),
            jnp.arange(0, jnp.pi, step=0.05),
            indexing="xy",
        ),
        axis=-1,
    )
    actions = agent.act(None, agent_params, positions, deterministic=True)[..., 0]  # [batch_size, 1] -> [batch_size]
    greedy_actions = GreedyPolicy(env).act(None, None, positions, deterministic=True)[..., 0]

    x_delta = positions[0, 1, 0] - positions[0, 0, 0]
    y_delta = positions[1, 0, 1] - positions[0, 0, 1]
    x_grid = positions[..., 0]
    y_grid = positions[..., 1]

    agent_x_vec = x_delta * env.actions[actions, 0] / 0.3
    agent_y_vec = y_delta * env.actions[actions, 1] / 0.3

    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].quiver(
        x_grid - agent_x_vec / 2.0,
        y_grid - agent_y_vec / 2.0,
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
    ax[0].set_aspect(1.0)

    greedy_x_vec = x_delta * env.actions[greedy_actions, 0] / 0.3
    greedy_y_vec = y_delta * env.actions[greedy_actions, 1] / 0.3

    ax[1].quiver(
        x_grid - greedy_x_vec / 2.0,
        y_grid - greedy_y_vec / 2.0,
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
    ax[1].set_aspect(1.0)

    plt.tight_layout()
    plt.show()


def main():
    args = tyro.cli(Args)
    key = jax.random.PRNGKey(args.random_seed)

    env = Environment()

    agent_key, key = jax.random.split(key)
    agent = PolicyGradientAgent(env.observation_shape, num_actions=env.num_actions, hidden_size=args.hidden_size)
    agent_params = agent.init_params(agent_key)

    optimizer = optax.adamw(learning_rate=args.learning_rate)
    opt_state = optimizer.init(agent_params)

    with tqdm(range(args.train_steps)) as pbar:
        for step in pbar:
            key, transitions = collect_transitions(
                key,
                env,
                agent,
                agent_params,
                num_envs=args.num_envs,
                num_steps=args.transitions_per_step,
            )
            agent_params, opt_state = agent.train(
                agent_params,
                opt_state,
                transitions,
                optimizer,
                discount=1.0,
            )

            if step % 10 == 0:
                eval_reward = evaluate_agent(env, agent, agent_params, num_episodes=20, seed=1337)
                pbar.set_description_str(
                    f"mean reward: {jnp.mean(transitions.reward):8.4f}, eval reward: {eval_reward:8.4f}"
                )

    with open(args.agent_save_path, "wb") as file:
        pickle.dump({"agent": agent, "agent_params": agent_params}, file)
    print(f'Trained agent saved to "{args.agent_save_path}"')

    # evaluate the agent's performance
    eval_reward = evaluate_agent(
        env,
        agent,
        agent_params,
        num_episodes=args.eval_num_episodes,
        seed=args.eval_random_seed,
        enable_progressbar=True,
    )

    eval_greedy = evaluate_agent(
        env,
        GreedyPolicy(env),
        None,
        num_episodes=args.eval_num_episodes,
        seed=args.eval_random_seed,
        enable_progressbar=True,
    )

    print(f"Average reward over {args.eval_num_episodes} episodes: {eval_reward:.4f}")
    print(f"Average reward of greedy policy: {eval_greedy:.4f}")

    visualize_policy(env, agent, agent_params)


if __name__ == "__main__":
    main()
