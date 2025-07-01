from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class State(NamedTuple):
    position: jnp.ndarray  # [float, float]
    accumulated_reward: jnp.ndarray  # [float]
    step_count: jnp.ndarray  # [int]
    done: jnp.ndarray  # [bool]


class Environment:
    def __init__(self):
        # actions are steps in 8 directions around the agent
        directions = jnp.array(
            [[x, y] for x in [-1.0, 0.0, 1.0] for y in [-1.0, 0.0, 1.0] if not (x == 0.0 and y == 0.0)]
        )
        # normalize directions to have a fixed size
        self.actions = 0.3 * directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
        self.bounds = jnp.array([2.0 * jnp.pi, jnp.pi])

    @property
    def num_actions(self) -> int:
        """
        Return the number of available actions.
        """
        return self.actions.shape[0]

    @property
    def observation_shape(self) -> tuple:
        """
        Return the shape of the observation space.
        """
        return (2,)

    def get_observation(self, state: State) -> jnp.ndarray:
        """
        Get the current observation of the agent's position.
        """
        return state.position

    def intermediate_reward(self, position: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the reward based on the agent's current position.
        """

        # penalize the agent for being outside the bounds
        return jnp.where(
            jnp.all((0.0 <= position) & (position <= self.bounds), axis=-1),
            jnp.maximum(0.0, jnp.sin(position[..., 0])),
            -10.0,
        )[..., None]  # ensure shape is [batch_size, 1]

    def get_reward(self, state: State) -> jnp.ndarray:
        """
        Get the reward for the current state.
        """
        # only return the accumulated reward if the episode is done
        return jnp.where(state.done, state.accumulated_reward, 0.0)

    def _reset_single(self, key: jnp.ndarray) -> State:
        """
        Reset the agent's position to a random point within the bounds.
        """
        y = jax.random.uniform(key, (), minval=0, maxval=jnp.pi)
        position = jnp.array([0.0, y])
        return State(
            position=position,
            accumulated_reward=jnp.array([0.0]),
            step_count=jnp.array([0]),
            done=jnp.array([False]),
        )

    @partial(jax.jit, static_argnames=("self", "num"))
    def reset(self, key: jnp.ndarray, num: int = 1) -> tuple[State, jnp.ndarray]:
        """
        Position agent randomly at the left edge of the box.

        :param key: JAX random key for generating new states.
        :param num: Number of agents to reset. Default is 1.
        """
        keys = jax.random.split(key, num)
        state = jax.vmap(self._reset_single)(keys)
        return state, self.get_observation(state)

    @partial(jax.jit, static_argnames=("self",))
    def masked_reset(self, key: jnp.ndarray, state: State, mask: jnp.ndarray) -> tuple[State, jnp.ndarray]:
        """
        Reset the agent's position if the mask is True.

        :param key: JAX random key for generating new states.
        :param state: The current state of the environment.
        :param mask: A boolean mask indicating which states to reset. (shape: [batch_size, 1])
        """
        batch, *_ = mask.shape
        keys = jax.random.split(key, batch)
        new_state = jax.vmap(self._reset_single)(keys)

        state = state._replace(
            position=jnp.where(mask, new_state.position, state.position),
            accumulated_reward=jnp.where(mask, new_state.accumulated_reward, state.accumulated_reward),
            step_count=jnp.where(mask, new_state.step_count, state.step_count),
            done=jnp.where(mask, new_state.done, state.done),
        )

        return state, self.get_observation(state)

    @partial(jax.jit, static_argnames=("self",))
    def step(self, state: State, actions: jnp.ndarray) -> tuple[State, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Move the agent in the specified direction and update the state.

        :param state: The current state of the environment.
        :param actions: The actions taken by the agent, which are indices into the action space (shape: [batch_size, 1])
        """

        new_positions = state.position + self.actions[actions[..., 0]]
        new_accumulated_reward = state.accumulated_reward + self.intermediate_reward(new_positions)
        new_step_count = state.step_count + 1
        done = new_step_count >= 10

        state = state._replace(
            position=new_positions,
            accumulated_reward=new_accumulated_reward,
            step_count=new_step_count,
            done=done,
        )

        return (
            state,
            self.get_observation(state),
            self.get_reward(state),
            state.done,
        )
