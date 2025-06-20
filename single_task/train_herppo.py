import sys
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)
from pathlib import Path
from typing import Optional, Tuple
import tyro
from dataclasses import dataclass, asdict
import wandb
import time
import random
import numpy as np
from tqdm import tqdm
import torch
from copy import copy
from dataclasses import dataclass, replace

import logging_callback
import lr_scheduler

import os
from mujoco_utils import composer_utils
import gymnasium as gym
from utils import get_env, make_envs

from HER import RolloutHindsightReplayBuffer, HERPPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import pickle
import shutil


@dataclass(frozen=True)
class Args:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "disabled"
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    deepmimic: bool = False
    mimic_task: str = "TwinkleTwinkleRousseau"
    midi_start_from: int = 0    
    residual_action: bool = False
    num_envs: int = 16
    pretrained: Optional[Path] = None
    initial_lr: float = 3e-4
    lr_decay_rate: float = 0.99
    residual_factor: float = 0.02
    n_steps: int = 512
    use_note_trajectory: bool = False
    mimic_z_axis: bool = False
    disable_hand_collisions: bool = True
    rsi: bool = False
    curriculum: bool = False
    total_iters: int = 1000

def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}

def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"HERPPO-{args.environment_name}-{args.seed}-{time.time()}"

    # Create experiment directory.
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True)

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # key=''
    # wandb.login(key=key)

    # wandb.init(
    #     project=args.project,
    #     config=asdict(args),
    #     name=run_name,
    #     sync_tensorboard=True,
    # )
    eval_args = copy(args)
    eval_args = replace(eval_args, rsi=False)
    eval_env = get_env(eval_args, record_dir=experiment_dir / "eval")
    def make_env():
        env = get_env(args)
        return Monitor(env)
    # Parallel environments
    vec_env = SubprocVecEnv([make_envs(make_env, i) for i in range(args.num_envs)], start_method="fork")

    lr_scheduler_instance = lr_scheduler.LR_Scheduler(initial_lr=args.initial_lr,
                                                      decay_rate=args.lr_decay_rate,)

    policy_kwargs = dict(activation_fn=torch.nn.GELU,
                     net_arch=dict(pi=[1024, 256], vf=[1024, 256]))
    model = HERPPO("MlpPolicy", 
                vec_env, 
                n_epochs=10,
                n_steps=args.n_steps,
                batch_size=1024,
                learning_rate=lr_scheduler_instance.lr_schedule,
                policy_kwargs=policy_kwargs, 
                verbose=2,
                tensorboard_log="./robopianist_rl/tensorboard/{}".format(run_name),
                rollout_buffer_class = RolloutHindsightReplayBuffer,
                )
    if args.pretrained is not None:
        # Reload learning rate scheduler
        custom_objects = { 'learning_rate': lr_scheduler_instance.lr_schedule}
        model = HERPPO.load(args.pretrained, env=vec_env, custom_objects=custom_objects)
    best_f1 = -np.inf
    # last_extending_curriculum_step = 0
    try:
        for i in range(args.total_iters):
            # Training
            model.learn(total_timesteps=args.n_steps*args.num_envs, 
                        progress_bar=True,
                        reset_num_timesteps=False,
                        callback= None)
            # Evaluation
            obs, info_init = eval_env.reset()
            # goal = obs[(15+15+10)*4:(15+15+10)*4+979*4].reshape(11,89,4).transpose(0,2,1)   # (11,4,89)
            # print(info_init['goal_state'][:5])
            # print('timestep t: ', goal[:5,3,:])
            # print('timestep t-3: ', goal[:5,0,:])
            # exit(1)           
            while True:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(action)
                # goal = obs[(15+15+10)*4:(15+15+10)*4+979*4].reshape(11,89,4).transpose(0,2,1)   # (11,4,89)
                # print(info['goal_state'][:5])
                # print('timestep t:\n', goal[:5,3,:])
                # print('timestep t-1:\n', goal[:5,2,:])
                # exit(1)  
                if done == True:
                    break
            log_dict = prefix_dict("eval", eval_env.env.get_statistics())
            music_dict = prefix_dict("eval", eval_env.env.get_musical_metrics())
            # wandb.log(log_dict | music_dict, step=i)
            # if args.deepmimic:
            #     wandb.log(prefix_dict("eval", eval_env.env.get_deepmimic_rews()), step=i)
            f1 = eval_env.env.get_musical_metrics()["f1"]
            if f1 > best_f1:
                print("best_f1:{}->{}".format(best_f1, eval_env.env.get_musical_metrics()["f1"]))
                best_f1 = eval_env.env.get_musical_metrics()["f1"]
                model.save("./robopianist_rl/ckpts/{}_best".format(run_name), exclude=['her_env'])
                video = wandb.Video(str(eval_env.env.latest_filename), fps=4, format="mp4")
                # wandb.log({"video": video, "global_step": i})
                try:
                    shutil.copy(str(eval_env.env.latest_filename), "./robopianist_rl/ckpts/{}.mp4".format(run_name))
                finally:
                    pass
            
            eval_env.env.latest_filename.unlink()  
    except KeyboardInterrupt:
        pass

    # model.save("./robopianist_rl/ckpts/{}".format(run_name))

    # Evaluate the trained model
    model = HERPPO.load("./robopianist_rl/ckpts/{}_best".format(run_name), env=vec_env)

    obs, _ = eval_env.reset()
    actions = []
    rewards = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, _, info = eval_env.step(action)
        rewards += reward
        if done:
            break
    print(f"Total reward: {rewards}")
    print(eval_env.env.latest_filename)
    print(eval_env.env.get_musical_metrics())
    actions = np.array(actions)
    np.save("./trained_songs/{}/actions_{}".format(args.mimic_task, args.mimic_task), actions)

    del model # remove to demonstrate saving and loading

if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))