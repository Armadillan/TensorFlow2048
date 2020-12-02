#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from env import PyEnv2048

FC_LAYER_PARAMS = (100, 100)
MAX_DURATION = 500

LEARNING_RATE = 1e-5
DISCOUNT_FACTOR = 0.9

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# Using huber loss instead of squared loss
LOSS_FN = common.element_wise_huber_loss

BUFFER_MAX_LEN = 500
BUFFER_BATCH_SIZE = 64
N_STEP_UPDATE = 1

INITAL_EPSILON = 1.0
END_EPSILON = 0.01
EPSILON_DECAY_STEPS = 10000

train_py_env = wrappers.TimeLimit(PyEnv2048(), duration=MAX_DURATION)
eval_py_env = wrappers.TimeLimit(PyEnv2048(), duration=MAX_DURATION)

train_env = tf_py_environment.TFPyEnvironment(PyEnv2048())
eval_env = tf_py_environment.TFPyEnvironment(PyEnv2048())

train_step_counter = tf.Variable(0)

num_episodes_metric = tf_metrics.NumberOfEpisodes(),
num_steps_metric = tf_metrics.EnvironmentSteps(),
avg_return_metric = tf_metrics.AverageReturnMetric(),
avg_episode_len_metric = tf_metrics.AverageEpisodeLengthMetric(),

train_metrics = [
            num_episodes_metric,
            num_steps_metric,
            avg_return_metric,
            avg_episode_len_metric
]

q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=FC_LAYER_PARAMS,
        activation_fn=tf.keras.activations.relu)

# Refactor to use tf.keras.optimizers.schedules.PolynomialDecay instead?
# Then it would have to be wrapped in a lambda
epsilon = tf.compat.v1.train.polynomial_decay(
    learning_rate=INITAL_EPSILON,
    global_step=train_step_counter,
    decay_steps=EPSILON_DECAY_STEPS,
    end_learning_rate=END_EPSILON
    )

agent = dqn_agent.DqnAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    n_step_update=N_STEP_UPDATE,
    q_network=q_net,
    optimizer=OPTIMIZER,
    epsilon_greedy=epsilon,
    td_errors_loss_fn=LOSS_FN,
    gamma=DISCOUNT_FACTOR,
    train_step_counter=train_step_counter
    )

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=BUFFER_MAX_LEN,
    )

dataset = replay_buffer.as_dataset(
    sample_batch_size=BUFFER_BATCH_SIZE,
    num_steps=N_STEP_UPDATE + 1,
    num_parallel_calls=3).prefetch(3)

dataset_iterator = iter(dataset)

agent.initialize()

agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)
