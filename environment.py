import jax
import jax.numpy as jnp


class Environment:
    def __init__(self):
        # actions are steps in 8 directions around the agent
        directions = jnp.array(
            [[x, y] for x in [-1.0, 0.0, 1.0] for y in [-1.0, 0.0, 1.0] if not (x == 0.0 and y == 0.0)]
        )
        # normalize directions to have a fixed size
        self.actions = 0.3 * directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
        self.bounds = jnp.array([2.0 * jnp.pi, jnp.pi])
        self.position = None
        self.accumulated_reward = 0.0
        self.step_count = 0

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

    def get_observation(self) -> jnp.ndarray:
        """
        Get the current observation of the agent's position.
        """
        return self.position

    def get_reward(self) -> float:
        """
        Calculate the reward based on the agent's current position.
        """
        if jnp.all((0.0 <= self.position) & (self.position <= self.bounds)):
            return jnp.maximum(0.0, jnp.sin(self.position[0])).item()

        # penalize the agent for being outside the bounds
        return -10.0

    def reset(self, key: jnp.ndarray) -> jnp.ndarray:
        """
        Position agent randomly at the left edge of the box.
        """
        y = jax.random.uniform(key, (), minval=0, maxval=jnp.pi)
        self.position = jnp.array([0.0, y])
        self.accumulated_reward = 0.0
        self.step_count = 0
        return self.get_observation()

    def step(self, action: int):
        """
        Move the agent in the specified direction.
        """
        self.position += self.actions[action]
        self.accumulated_reward += self.get_reward()
        self.step_count += 1

        terminated = self.step_count >= 10
        observation = self.get_observation()
        reward = self.accumulated_reward if terminated else 0.0

        return observation, reward, terminated
