import pickle
from dataclasses import dataclass

import tyro

from environment import Environment
from utils import evaluate_agent


@dataclass
class Args:
    agent_path: str = "agent.pkl"

    num_eval_episodes: int = 1000
    random_seed: int = 1234


def main():
    args = tyro.cli(Args)

    env = Environment()

    with open(args.agent_path, "rb") as file:
        checkpoint = pickle.load(file)

    agent = checkpoint["agent"]
    agent_params = checkpoint["agent_params"]

    average_reward = evaluate_agent(
        env, agent, agent_params, num_episodes=args.num_eval_episodes, seed=args.random_seed, enable_progressbar=True
    )
    print(f"Average reward over {args.num_eval_episodes} episodes: {average_reward:.2f}")


if __name__ == "__main__":
    main()
