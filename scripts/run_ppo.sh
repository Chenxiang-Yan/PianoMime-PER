#!/bin/bash

source .env
# echo "WANDB_API_KEY: $WANDB_API_KEY"
export WANDB_BASE_URL=https://api.bandw.top
wandb login --relogin $WANDB_API_KEY

WANDB_DIR=/tmp/robopianist/ XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0  \
python PianoMime-PER/single_task/train_ppo.py \
    --root-dir /tmp/robopianist/rl/ \
    --warmstart-steps 5000 \
    --max-steps 1000000 \
    --discount 0.99 \
    --trim-silence \
    --gravity-compensation \
    --control-timestep 0.05 \
    --n-steps-lookahead 0 \
    --disable_fingering_reward \
    --disable_hand_collisions \
    --disable_forearm_reward \
    --tqdm-bar \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --midi-start-from 0 \
    --residual-action \
    --frame-stack 4 \
    --num-envs 32 \
    --initial-lr 3e-4 \
    --lr-decay-rate 0.999 \
    --n-steps 512 \
    --mimic-task "Petrunko_3" \
    --environment-name "Petrunko_3" \
    --use-note-trajectory \
    --total-iters 2000 \
    --residual-factor 0.03 \
    --deepmimic \
    