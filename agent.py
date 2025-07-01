from abc import ABC, abstractmethod
from functools import partial, reduce
from operator import mul
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

type Params = dict[str, jnp.ndarray]


class Transition(NamedTuple):
    observation: jnp.ndarray  # [float]
    action: jnp.ndarray  # [int]
    reward: jnp.ndarray  # [float]
    next_observation: jnp.ndarray  # [float]
    done: jnp.ndarray  # [bool]


class BaseAgent(ABC):
    @abstractmethod
    def act(
        self, key: jnp.ndarray, params: Params, observation: jnp.ndarray, deterministic: bool = False
    ) -> int | jnp.ndarray:
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
    def train(self, params: Params, transitions: Transition, **kwargs):
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

    def __init__(self, observation_shape: tuple, num_actions: int = 8, hidden_size: int = 128):
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.hidden_size = hidden_size

    def init_params(self, key: jnp.ndarray) -> Params:
        """
        Initialize the parameters of the agent's policy network.
        """
        input_features = reduce(mul, self.observation_shape, 1)
        w1_key, w2_key = jax.random.split(key, num=2)

        initializer = jax.nn.initializers.glorot_normal()

        return {
            "w1": initializer(w1_key, (input_features, self.hidden_size)),
            "b1": jnp.zeros((self.hidden_size,)),
            "w2": initializer(w2_key, (self.hidden_size, self.num_actions)),
            "b2": jnp.zeros((self.num_actions,)),
        }

    @partial(jax.jit, static_argnames=("self",))
    def _forward(self, params: Params, observation: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the neural network to compute action logits.
        """
        hidden = jax.nn.gelu(jnp.matmul(observation, params["w1"]) + params["b1"])
        action_logits = jnp.matmul(hidden, params["w2"]) + params["b2"]
        return action_logits

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def act(
        self, key: jnp.ndarray, params: Params, observation: jnp.ndarray, deterministic: bool = False
    ) -> int | jnp.ndarray:
        """
        Select an action based on the current observation. Output can be a single action or a batch of actions
        depending on the shape of the observation.
        """
        action_logits = self._forward(params, observation)
        if deterministic:
            return jnp.argmax(action_logits, axis=-1, keepdims=True)  # [batch_size, 1]

        # add a gumbel noise to the logits to improve exploration
        gumbel_noise = jax.random.gumbel(key, shape=action_logits.shape)
        return jax.random.categorical(key, action_logits + gumbel_noise, axis=-1)[..., None]  # [batch_size, 1]

    def _policy_loss(
        self,
        params: Params,
        transitions: Transition,
        advantage: jnp.ndarray,
        entropy_factor: float = 1e-3,
    ) -> jnp.ndarray:
        batch_size, *_ = advantage.shape

        action_logits = self._forward(params, transitions.observation)
        action_log_probs = jax.nn.log_softmax(action_logits, axis=-1)  # [batch_size, num_actions]

        # entropy penalty to encourage exploration
        entropy_penalty = -jnp.mean(jnp.sum(jnp.exp(action_log_probs) * action_log_probs, axis=-1))

        selected_action_log_probs = action_log_probs[jnp.arange(batch_size), transitions.action[..., 0]]  # [batch_size]
        actor_loss = -jnp.mean(selected_action_log_probs * advantage[..., 0])  # [batch_size]
        # minimizing this loss will encourage actions that have positive advantage and discourage
        # those with negative advantage
        return actor_loss + entropy_factor * entropy_penalty

    @partial(jax.jit, static_argnames=("self", "entropy_factor"))
    def _policy_gradient(
        self,
        params: Params,
        transitions: Transition,
        rewards: jnp.ndarray,
        entropy_factor: float = 1e-3,
    ) -> jnp.ndarray:
        return jax.grad(self._policy_loss)(params, transitions, rewards, entropy_factor)

    @partial(jax.jit, static_argnames=("self", "discount", "optimizer", "entropy_factor"))
    def train(
        self,
        params: Params,
        opt_state: optax.OptState,
        transitions: Transition,
        optimizer: optax.GradientTransformationExtraArgs,
        discount: float = 1.0,
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
            jnp.array([0.0]),
            transitions,
            reverse=True,
        )

        normalized_rewards = (cumulative_rewards - jnp.mean(cumulative_rewards)) / (jnp.std(cumulative_rewards) + 1e-8)
        params_grad = self._policy_gradient(params, transitions, normalized_rewards, entropy_factor)

        updates, opt_state = optimizer.update(params_grad, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        return updated_params, opt_state


class GreedyPolicy(BaseAgent):
    """
    A simple greedy policy that selects actions based on the maximum reward accessible from the current observation.
    This policy cannot be trained and is used for comparison purposes. Exploits the environment's structure which is
    not accessible to the agent during training.
    """

    def __init__(self, environment: "Environment"):
        self.environment = environment

    def act(self, key: jnp.ndarray, params: Params, observation: jnp.ndarray, deterministic: bool = True) -> int:
        """
        Return the action that maximizes the reward based on the current observation.
        """
        next_observation = observation[..., None, :] + self.environment.actions  # [..., num_actions, 2]
        rewards = jnp.where(
            jnp.all((0.0 <= next_observation) & (next_observation <= self.environment.bounds), axis=-1),
            jnp.maximum(0.0, jnp.sin(next_observation[..., 0])),
            -10.0,
        )  # [..., num_actions]
        return jnp.argmax(rewards, axis=-1, keepdims=True)  # [..., 1]

    def train(self, params: Params, transitions: Transition, **kwargs):
        """
        This policy cannot be trained, so this method is a no-op.
        """
        pass
