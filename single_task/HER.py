import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from typing import Optional, Tuple, Union, Any, TypeVar
from collections.abc import Generator
from stable_baselines3.common.type_aliases import RolloutBufferSamples, GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from stable_baselines3.ppo import PPO
from gymnasium import spaces
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from dm_control.utils.rewards import tolerance

SelfHERPPO = TypeVar("SelfHERPPO", bound="HERPPO")
 
class RolloutHindsightReplayBuffer(RolloutBuffer):
    def __init__(
        self, 
        buffer_size, 
        observation_space, 
        action_space, 
        **kwargs
    ):
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        self.replay_k = 4
        self.future_p = 1 - (1.0 / (1 + self.replay_k))     # prob. of replace original sample with HER sample, here p=0.8
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        mimic_reward: np.ndarray,
        curr_goal: np.ndarray,
        curr_piano_and_sustain: np.ndarray,
        piano_activation: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)

        ##### our modification #####
        self.mimic_rewards[self.pos] = np.array(mimic_reward)
        self.current_goal[self.pos] = np.array(curr_goal)
        self.current_piano_and_sustain[self.pos] = np.array(curr_piano_and_sustain)
        self.piano_activation[self.pos] = np.array(piano_activation)
        ############################

        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    ##### our modification #####
    def reset(self) -> None:
        super().reset()
        self.mimic_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.current_goal = np.zeros((self.buffer_size, self.n_envs, 89), dtype=np.float32)
        self.current_piano_and_sustain = np.zeros((self.buffer_size, self.n_envs, 89), dtype=np.float32)
        self.piano_activation = np.zeros((self.buffer_size, self.n_envs, 88), dtype=np.float32)

    def generate_piano_and_sustain_state(self, current_piano_and_sustain):
        piano_sustain_obs = np.zeros((self.buffer_size, self.n_envs, 89, 4), dtype=np.float32)
        zero_padding = np.zeros((3, self.n_envs, 89), dtype=np.float32)
        piano_and_sustain = np.concatenate((zero_padding, current_piano_and_sustain), axis=0)  # (buffer_size+3, n_envs, 89)
        for i in range(self.buffer_size):
            piano_sustain_obs[i] = piano_and_sustain[i:i+4].transpose(1,2,0)    # (n_envs, 89, 4)
        return piano_sustain_obs    # (buffer_size, n_envs, 89, 4)

    def generate_her_goal(self, current_piano_and_sustain):
        her_goal = np.zeros((self.buffer_size, self.n_envs, 979*4), dtype=np.float32)     # (88+1)*(10+1)*4, n_piano_states=88, n_step_lookahead=10, n_framestack=4

        piano_sustain_obs = self.generate_piano_and_sustain_state(current_piano_and_sustain)
        zero_padding = np.zeros((10-1, self.n_envs, 89, 4), dtype=np.float32)   # n_step_lookahead=10
        piano_sustain_obs = np.concatenate((piano_sustain_obs, zero_padding), axis=0)   # (buffer_size+9, n_envs, 89, 4)

        g_t = self.observations[:, :, (15+15+10)*4:(15+15+10)*4+89*4]        # (buffer_size, n_envs, 89*4)

        for i in range(self.buffer_size):
            temp = np.concatenate(tuple(piano_sustain_obs[i:i+10]), axis=-2).reshape(self.n_envs, -1)  # (n_envs, 89*10*4) 
            her_goal[i] = np.concatenate((g_t[i], temp), axis=-1)    # (n_envs, 89*11*4)
        
        return her_goal         # (buffer_size, n_envs, 89*11*4) 

    def generate_her_indices_and_offset(self):
        her_indices = np.where(np.random.uniform(size=self.buffer_size*self.n_envs) < self.future_p)[0]
        buffer_idxs, env_idxs = her_indices // self.n_envs, her_indices % self.n_envs
        offset_idxs = np.random.randint(buffer_idxs, self.buffer_size, size=her_indices.shape)
        return buffer_idxs, env_idxs, offset_idxs
    
    def her_rew_func(self, buffer_idxs, env_idxs, offset_idxs):
        K = len(buffer_idxs)    # batch size
        _KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05
        goal = self.current_piano_and_sustain[offset_idxs, env_idxs, :]         # (K, 89)
        actual = self.current_piano_and_sustain[buffer_idxs, env_idxs, :]       # (K, 89)
        piano_activation = self.piano_activation[buffer_idxs, env_idxs, :]      # (K, 88)
        
        # sustain
        sustain_rew =  tolerance(
                            goal[:, -1] - actual[:, -1],
                            bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
                            margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
                            sigmoid="gaussian",
                        )
        
        # key press
        key_press_rew = np.zeros((K,), dtype=np.float32)
        rews = tolerance(
            goal[:, :-1] - actual[:, :-1],
            bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
            margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
            sigmoid="gaussian",
        )
        for i in range(K):
            on = np.flatnonzero(goal[i, :-1])   # (88,) -> (n_on,)
            if on.size > 0:
                key_press_rew[i] += 0.5 * rews[i, on].mean()
            # If there are any false positives, the remaining 0.5 reward is lost.
            off = np.flatnonzero(1 - goal[i, :-1])
            key_press_rew[i] += 0.5 * (1 - float(piano_activation[i, off].any()))

        return sustain_rew + 2 * key_press_rew

    def process_her_obs_and_rew(self, her_goals, buffer_idxs, env_idxs, offset_idxs, rew_func):
        assert len(buffer_idxs) == len(env_idxs)
        self.observations[buffer_idxs, env_idxs, (15+15+10)*4:(15+15+10)*4+979*4] = her_goals[offset_idxs, env_idxs, :]
        self.rewards[buffer_idxs, env_idxs] = rew_func(buffer_idxs, env_idxs, offset_idxs) + self.mimic_rewards[buffer_idxs, env_idxs]
    ############################
    
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""

        ##### our modification #####
        if not self.generator_ready:
            goals = self.generate_her_goal(self.current_piano_and_sustain)
            buffer_idxs, env_idxs, offset_idxs = self.generate_her_indices_and_offset()
            self.process_her_obs_and_rew(goals, buffer_idxs, env_idxs, offset_idxs, self.her_rew_func)
        ############################

        indices = np.random.permutation(self.buffer_size * self.n_envs)
        
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
        
        # after the "Prepare the data" step, the shape of buffer data are:
        # self.observations: (n_envs*buffer_size, 4832(obs_dim))
        # self.actions: (n_envs*buffer_size, 47(action_dim))
        # self.values: (n_envs*buffer_size, 1)
        # self.log_probs: (n_envs*buffer_size, 1)
        # self.advantages: (n_envs*buffer_size, 1)
        # self.returns: (n_envs*buffer_size, 1)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    

class HERPPO(PPO):
    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        rollout_buffer_class : Optional[type[RolloutBuffer]] = RolloutBuffer, # RolloutHindsightReplayBuffer,
        **kwargs,
    ):        
        super().__init__(policy, env, rollout_buffer_class= rollout_buffer_class, **kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        index = 0
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            ##### our modification #####
            mimic_rewards = np.array([infos[i]['her_mimic_reward'] for i in range(len(infos))])
            curr_goal = np.array([infos[i]['current_goal'] for i in range(len(infos))])
            curr_piano_and_sustain = np.array([infos[i]['current_piano_and_sustain'] for i in range(len(infos))])
            piano_activation = np.array([infos[i]['piano_activation'] for i in range(len(infos))])
            ############################

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
                    ##### our modification #####
                    mimic_rewards[idx] += self.gamma * terminal_value
                    ############################

            rollout_buffer.add(     ###TODO: overload rollout_buffer.add
                self._last_obs,  # type: ignore[arg-type]; ag
                actions,
                rewards,
                ##### our modification #####
                mimic_rewards,
                curr_goal,
                curr_piano_and_sustain,
                piano_activation,
                ############################
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

import robopianist.wrappers as robopianist_wrappers
class Dm2GymInfoWrapper(robopianist_wrappers.Dm2GymWrapper):
    def __init__(self, environment):
        super().__init__(environment)
    
    def _unwrap_env(self, env):
        while hasattr(env, "env"):
            env = env.env
        return env
    
    def step(self, action):
        timestep = self.env.step(action)
        observation = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        info = {}
        
        base_env = self._unwrap_env(self.env)
        task = base_env.task
        if hasattr(task, "_her_mimic_reward"):
            info['her_mimic_reward'] = task._her_mimic_reward
        if hasattr(task, "_goal_state"):
            info['current_goal'] = task._goal_state[0]  # (89,)
        if hasattr(task, "_piano_state") and hasattr(task, "_sustain_state"):
            info['current_piano_and_sustain'] = np.concatenate((task._piano_state, task._sustain_state.reshape(1)))
        if hasattr(task, '_piano_activation'):
            info['piano_activation'] = task._piano_activation

        return observation, reward, done, False, info
        
import dm_env_wrappers as wrappers
import tree
def _concat(values) -> np.ndarray:
    leaves = list(map(np.atleast_1d, tree.flatten(values)))
    return np.concatenate(leaves)
class ViewObsConcatObservationWrapper(wrappers.ConcatObservationWrapper):
    def __init__(self, environment, name_filter = None):
        super().__init__(environment, name_filter)

    def _convert_observation(self, observation):
        obs = {k: observation[k] for k in self._obs_names}

        obs_info = {}
        for key, value in obs.items():
            obs_info[key] = value.shape
        self._obs_info = obs_info

        # def unwrap_env(env):
        #     while hasattr(env, "env"):
        #         env = env.env
        #     return env
        
        # base_env = unwrap_env(self.env)
        # base_env.task._obs_info = obs_info

        return _concat(obs)