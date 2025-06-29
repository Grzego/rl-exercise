from abc import ABC, abstractmethod
from functools import partial, reduce
from operator import mul
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    observation: jnp.ndarray  # [float]
    action: jnp.ndarray  # [int]
    reward: jnp.ndarray  # [float]
    next_observation: jnp.ndarray  # [float]
    done: jnp.ndarray  # [bool]


class BaseAgent(ABC):
    @abstractmethod
    def act(self, key: jnp.ndarray, observation: jnp.ndarray, deterministic: bool = False) -> int | jnp.ndarray:
        """
        Given an observation, return an action.

        :param key: JAX random key for stochastic actions.
        :param observation: The current observation from the environment.
        :param deterministic: If True, select the action deterministically (e.g., argmax of action logits).
                              If False, select a stochastic action.

        :return: The selected action as an integer.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, transitions: Transition, **kwargs):
        """
        Train the agent using collected transitions.

        :param transitions: A batch of transitions containing observations, actions, rewards, next observations, and
            done flags.
        :param kwargs: Additional parameters for training, such as discount factor.
        """
        raise NotImplementedError()


class PolicyGradientAgent(BaseAgent):
    """
    A simple policy gradient agent that uses a neural network to approximate the policy.
    """

    def __init__(self, key: jnp.ndarray, observation_shape: tuple, num_actions: int = 8, hidden_size: int = 128):
        self.num_actions = num_actions
        input_features = reduce(mul, observation_shape, 1)

        w1_key, w2_key = jax.random.split(key, num=2)
        self.params = {
            "w1": 1e-1 * jax.random.normal(w1_key, (input_features, hidden_size)),
            "b1": jnp.zeros((hidden_size,)),
            "w2": 1e-1 * jax.random.normal(w2_key, (hidden_size, num_actions)),
            "b2": jnp.zeros((num_actions,)),
        }

    @partial(jax.jit, static_argnames=("self",))
    def _forward(self, params: dict[str, jnp.ndarray], observation: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the neural network to compute action logits.
        """
        is_single_observation = observation.ndim == 1
        observation = observation[None, ...] if is_single_observation else observation
        hidden = jax.nn.gelu(jnp.matmul(observation, params["w1"]) + params["b1"])
        logits = jnp.matmul(hidden, params["w2"]) + params["b2"]
        return logits[0] if is_single_observation else logits

    def act(self, key: jnp.ndarray, observation: jnp.ndarray, deterministic: bool = False) -> int | jnp.ndarray:
        """
        Select an action based on the current observation. Output can be a single action or a batch of actions
        depending on the shape of the observation.
        """
        action_logits = self._forward(self.params, observation)
        if deterministic:
            return jnp.argmax(action_logits, axis=-1)

        # add a gumbel noise to the logits to improve exploration
        gumbel_noise = jax.random.gumbel(key, shape=action_logits.shape)
        return jax.random.categorical(key, action_logits + gumbel_noise, axis=-1)

    def _policy_loss(
        self,
        params: dict[str, jnp.ndarray],
        transitions: Transition,
        advantage: jnp.ndarray,
        entropy_factor: float = 1e-3,
    ) -> jnp.ndarray:
        batch_size, *_ = advantage.shape

        action_log_probs = jax.nn.log_softmax(
            self._forward(params, transitions.observation), axis=-1
        )  # [batch_size, num_actions]

        # entropy penalty to encourage exploration
        entropy_penalty = -jnp.mean(jnp.sum(jnp.exp(action_log_probs) * action_log_probs, axis=-1))

        selected_action_log_probs = action_log_probs[jnp.arange(batch_size), transitions.action]  # [batch_size]
        # minimizing this loss will encourage actions that have positive advantage and discourage
        # those with negative advantage
        return -jnp.mean(advantage * selected_action_log_probs) + entropy_factor * entropy_penalty

    @partial(jax.jit, static_argnames=("self", "entropy_factor"))
    def _policy_gradient(
        self,
        params: dict[str, jnp.ndarray],
        transitions: Transition,
        advantage: jnp.ndarray,
        entropy_factor: float = 1e-3,
    ) -> jnp.ndarray:
        return jax.grad(self._policy_loss)(params, transitions, advantage, entropy_factor)

    @partial(jax.jit, static_argnames=("self", "discount", "learning_rate", "entropy_factor"))
    def _train(
        self,
        params: dict[str, jnp.ndarray],
        transitions: Transition,
        discount: float = 1.0,
        learning_rate: float = 1e-3,
        entropy_factor: float = 1e-3,
    ):
        """
        Compute the policy gradient and update the parameters.

        :param params: Current parameters of the agent.
        :param transitions: A batch of transitions containing observations, actions, rewards, next observations, and done flags.
        :return: Updated parameters after applying the policy gradient.
        """

        def _discount_rewards(carry: jnp.ndarray, transition: Transition):
            new_carry = carry * discount * (1.0 - transition.done) + transition.reward
            return new_carry, new_carry

        _, cumulative_rewards = jax.lax.scan(
            _discount_rewards,
            0.0,
            transitions,
            reverse=True,
        )

        # use cumulative rewards as advantage
        advantage = cumulative_rewards
        params_grad = self._policy_gradient(params, transitions, advantage, entropy_factor)

        # update parameters using the computed gradients (simple SGD update)
        updated_params = jax.tree.map(
            lambda p, g: p - learning_rate * g,
            params,
            params_grad,
        )
        return updated_params

    def train(
        self, transitions: Transition, discount: float = 1.0, learning_rate: float = 1e-3, entropy_factor: float = 1e-3
    ):
        # clip rewards to avoid large updates and stabilize training
        transitions = transitions._replace(reward=jnp.clip(transitions.reward, -10.0, 10.0))

        self.params = self._train(
            self.params,
            transitions,
            discount=discount,
            learning_rate=learning_rate,
            entropy_factor=entropy_factor,
        )


class GreedyPolicy(BaseAgent):
    """
    A simple greedy policy that selects actions based on the maximum reward accessible from the current observation.
    This policy cannot be trained and is used for comparison purposes. Exploits the environment's structure which is
    not accessible to the agent during training.
    """

    def __init__(self, environment: "Environment"):
        self.environment = environment

    def act(self, key: jnp.ndarray, observation: jnp.ndarray, deterministic: bool = True) -> int:
        """
        Return the action that maximizes the reward based on the current observation.
        """
        next_observation = observation[..., None, :] + self.environment.actions  # [..., num_actions, 2]
        rewards = jnp.where(
            jnp.all((0.0 <= next_observation) & (next_observation <= self.environment.bounds), axis=-1),
            jnp.maximum(0.0, jnp.sin(next_observation[..., 0])),
            -10.0,
        )  # [..., num_actions]
        return jnp.argmax(rewards, axis=-1)

    def train(self, transitions: Transition, **kwargs):
        """
        This policy cannot be trained, so this method is a no-op.
        """
        pass
