#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

# importing my environment
from env import PyEnv2048

environment = PyEnv2048()

# Works both as a python and tf environment
environment = tf_py_environment.TFPyEnvironment(environment)

up = np.array(0, dtype=np.int8)
right = np.array(1, dtype=np.int8)
down = np.array(2, dtype=np.int8)
left = np.array(3, dtype=np.int8)

total_reward = 0

time_step = environment.reset()

actions_n = []

for _ in range(30):
    actions = 0
    time_step = environment.reset()
    while not time_step.is_last():

        for action in (up, right, down, left):
            time_step = environment.step(action)
            total_reward += time_step.reward
            actions += 1
            if time_step.is_last():
                break
    actions_n.append(actions)

print("Average reward:")
print(total_reward.numpy()[0]/30)
print("Average steps:")
print(sum(actions_n)/len(actions_n))
