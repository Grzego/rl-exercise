
# RL Exercise

A simple reinforcement learning agent using policy gradients in a custom 2D environment.

## Requirements

Make sure you have `uv` installed. You can find installation instructions in the [uv documentation](https://github.com/astral-sh/uv#installation).
On the first run, `uv` will automatically create a virtual environment and install all required packages.

### To train and evaluate an agent

```bash
uv run main.py
```

Example output:
```bash
mean reward:  -0.17, eval reward:   8.28: 100%|████████████████████| 500/500 [01:05<00:00,  7.59it/s]
Trained agent saved to "agent.pkl"
evaluating agent: 100%|█████████████████████████████████████████| 1000/1000 [00:03<00:00, 304.20it/s]
Average reward over 1000 episodes: 8.18
```

or to run the evaluation only (after training):

```bash
uv run evaluate.py --agent-path=agent.pkl
```

Example output:
```bash
evaluating agent: 100%|█████████████████████████████████████████| 1000/1000 [00:03<00:00, 275.96it/s]
Average reward over 1000 episodes: 8.18
```

