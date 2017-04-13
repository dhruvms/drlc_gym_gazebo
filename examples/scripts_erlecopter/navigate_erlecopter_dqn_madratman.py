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
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
import gym

def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari SpaceInvaders')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--mode', default='vanilla', type=str, help='vanilla or double dqn')

    args.env = 'GazeboErleCopterNavigate-v0'
    args = parser.parse_args()
    print " MODE IS", args.mode

    video_every_nth = 50000
    eval_every_nth = 50000
        
    agent = DQNAgent(env=args.env, gamma=0.99, target_update_freq=10000, num_burn_in=50000, train_freq=4, batch_size=32, mode=args.mode)
    agent.fit(num_iterations = int(5e6), max_episode_length=100000, save_model_every_nth=10000, eval_every_nth=eval_every_nth, log_loss_every_nth=1000, video_every_nth=video_every_nth)

if __name__ == '__main__':
    main()
