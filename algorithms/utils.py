from typing import Dict, List, Any, NamedTuple
import numpy as np
import torch
from gymnasium import Space, spaces
from stable_baselines3.common.buffers import ReplayBuffer as RB
from stable_baselines3.common.vec_env import VecNormalize

class ExtendedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    latent_actions: torch.Tensor
    previous_latent_actions: torch.Tensor

class ExtendedReplayBufferSamples2(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    latent_actions: torch.Tensor
    previous_latent_actions: torch.Tensor
    latent_policy_actions: torch.Tensor


class ReplayBuffer(RB):
    def __init__(self, buffer_size: int, observation_space: Space, action_space: Space, latent_action_dim: int, device: torch.device | str = "auto", n_envs: int = 1, optimize_memory_usage: bool = False, handle_timeout_termination: bool = True, add_latent_policy_actions: bool = False):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        self.latent_action_dim = latent_action_dim
        self.latent_actions = np.zeros(
            (self.buffer_size, self.n_envs, latent_action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.previous_latent_actions = np.zeros(
            (self.buffer_size, self.n_envs, latent_action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.add_latent_policy_actions = add_latent_policy_actions
        if add_latent_policy_actions:
            self.latent_policy_actions = np.zeros(
                (self.buffer_size, self.n_envs, latent_action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
            )

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]], latent_action: np.ndarray, previous_latent_action: np.ndarray, latent_policy_actions: np.ndarray | None = None) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        latent_action = latent_action.reshape((self.n_envs, self.latent_action_dim))
        self.latent_actions[self.pos] = np.array(latent_action)
        previous_latent_action = previous_latent_action.reshape((self.n_envs, self.latent_action_dim))
        self.previous_latent_actions[self.pos] = np.array(previous_latent_action)
        if self.add_latent_policy_actions:
            latent_policy_actions = latent_policy_actions.reshape((self.n_envs, self.latent_action_dim))
            self.latent_policy_actions[self.pos] = np.array(latent_policy_actions)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> ExtendedReplayBufferSamples | ExtendedReplayBufferSamples2:
        return super().sample(batch_size, env)

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> ExtendedReplayBufferSamples | ExtendedReplayBufferSamples2:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = [
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.latent_actions[batch_inds, env_indices, :],
            self.previous_latent_actions[batch_inds, env_indices, :],
        ]
        if self.add_latent_policy_actions:
            data.append(
                self.latent_policy_actions[batch_inds, env_indices, :]
            )
            return ExtendedReplayBufferSamples2(*tuple(map(self.to_torch, data)))
        else:
            return ExtendedReplayBufferSamples(*tuple(map(self.to_torch, data)))
