#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam

import deeprl_hw2 as tfrl

from deeprl_hw2.objectives import *
import gym

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir

def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari SpaceInvaders')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--mode', default='vanilla', type=str, help='vanilla or double dqn')
    parser.add_argument('--question', default='deep', type=str, help='q2, q3, q4, deep, q7, eval_table')

    args = parser.parse_args()
    print " MODE IS", args.mode

    if args.question=="q2":
        from deeprl_hw2.dqn_q2 import DQNAgent
    elif args.question=="q3":
        from deeprl_hw2.dqn_q3 import DQNAgent
    elif args.question=="q4":
        from deeprl_hw2.dqn_q4 import DQNAgent
    elif args.question=="q7":
        from deeprl_hw2.dqn_q7 import DQNAgent
    elif args.question=="eval_table":
        from deeprl_hw2.evaluation_table import DQNAgent
    else:
        from deeprl_hw2.dqn import DQNAgent

    video_every_nth = 50000
    eval_every_nth = 50000

    if args.env == "breakout":
        args.env = 'Breakout-v0'
        video_every_nth = 50000
    if args.env == "space_invaders":
        args.env = 'SpaceInvaders-v0'
    if args.env == 'enduro':
        args.env = 'Enduro-v0'
        video_every_nth = 50000
        eval_every_nth = 50000        
        
    agent = DQNAgent(env=args.env, gamma=0.99, target_update_freq=10000, num_burn_in=50000, train_freq=4, batch_size=32, mode=args.mode)
    agent.fit(num_iterations = int(5e6), max_episode_length=100000, save_model_every_nth=10000, eval_every_nth=eval_every_nth, log_loss_every_nth=1000, video_every_nth=video_every_nth)

if __name__ == '__main__':
    main()
