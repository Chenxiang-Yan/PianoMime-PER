import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from typing import Optional, Tuple, Union, Any
from collections.abc import Generator
from stable_baselines3.common.type_aliases import RolloutBufferSamples, GymEnv, Schedule
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.ppo import PPO
from gymnasium import spaces
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.policies import ActorCriticPolicy

class RolloutPrioritizedReplayBuffer(RolloutBuffer):
    def __init__(
        self, 
        buffer_size, 
        observation_space, 
        action_space, 
        device = "auto", 
        gae_lambda = 1, 
        gamma = 0.99, 
        n_envs = 1
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self.alpha = 0.7
        self.beta = 0.4
        self.eps = 0.01
    
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""

        # indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
        
        ##### OUR MODIFICATION #####
        # after the "Prepare the data" step, the shape of buffer data are:
        # self.observations: (n_envs*buffer_size, 4832(obs_dim))
        # self.actions: (n_envs*buffer_size, 47(action_dim))
        # self.values: (n_envs*buffer_size, 1)
        # self.log_probs: (n_envs*buffer_size, 1)
        # self.advantages: (n_envs*buffer_size, 1)
        # self.returns: (n_envs*buffer_size, 1)

        self.p = np.abs(self.advantages.flatten())       # (n_envs*buffer_size, ), the order of 0-th dimension is (n_envs, buffer_size) 
                                                    # i.e. different envs at the same timestep are gathered
        self.p = (self.p + self.eps) ** self.alpha
        self.p /= self.p.sum()

        ############################

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            ##### OUR MODIFICATION #####

            indices = np.random.choice(self.n_envs*self.buffer_size, size=batch_size, p=self.p, replace=True)
            yield self._get_samples(indices)

            ############################
            # yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> Tuple[RolloutBufferSamples, np.ndarray]:

        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        ##### OUR MODIFICATION #####

        priorities = self.p[batch_inds]
        return RolloutBufferSamples(*tuple(map(self.to_torch, data))), priorities
        
        ############################

        # return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

class PERPPO(PPO):
    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):        
        super().__init__(policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, normalize_advantage, ent_coef, vf_coef, max_grad_norm, use_sde, sde_sample_freq, rollout_buffer_class, rollout_buffer_kwargs, target_kl, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)
       
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer

            # for rollout_data in self.rollout_buffer.get(self.batch_size):

            ##### OUR MODIFICATION #####

            for rollout_data, priorities in self.rollout_buffer.get(self.batch_size):
                weight_per = (1 / self.rollout_buffer.buffer_size / priorities) ** self.rollout_buffer.beta
                weight_per /= np.max(weight_per)    
                weight_per = th.as_tensor(weight_per).to(self.device)   # (batch_size,)

                ############################
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                ##### OUR MODIFICATION #####

                policy_loss = -(weight_per * th.min(policy_loss_1, policy_loss_2)).mean()

                ############################

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # value_loss = F.mse_loss(rollout_data.returns, values_pred)
                ##### OUR MODIFICATION #####

                value_loss = (weight_per * (rollout_data.returns - values_pred) ** 2).mean()

                ############################
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)