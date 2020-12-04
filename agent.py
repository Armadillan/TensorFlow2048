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
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from env import PyEnv2048, PyEnv2048FlatObservations

FC_LAYER_PARAMS = (64, 32)
MAX_DURATION = 500

LEARNING_RATE = 1e-7
# gamma
DISCOUNT_FACTOR = 0.95

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

LOSS_FN = common.element_wise_squared_loss

BUFFER_MAX_LEN = 500
BUFFER_BATCH_SIZE = 64
N_STEP_UPDATE = 3

COLLECTION_STEPS = 1
NUM_EVAL_EPISODES = 10
NUM_TRAINING_ITERATIONS = 1000000

INITIAL_EPSILON = 0.99
END_EPSILON = 0.01
EPSILON_DECAY_STEPS = 900000

PUNISHMENT_FOR_BAD_MOVES = 32

LOG_INTERVAL = 200
EVAL_INTERVAL = 5000

train_py_env = wrappers.TimeLimit(PyEnv2048FlatObservations(
    PUNISHMENT_FOR_BAD_MOVES),
                                  duration=MAX_DURATION)
eval_py_env = wrappers.TimeLimit(PyEnv2048FlatObservations(
    PUNISHMENT_FOR_BAD_MOVES),
                                 duration=MAX_DURATION)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

train_step_counter = tf.Variable(0)

q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=FC_LAYER_PARAMS,
        activation_fn=tf.keras.activations.relu)

# Refactor to use tf.keras.optimizers.schedules.PolynomialDecay instead?
# Then it would have to be wrapped in a lambda
epsilon = tf.compat.v1.train.polynomial_decay(
    learning_rate=INITIAL_EPSILON,
    global_step=train_step_counter,
    decay_steps=EPSILON_DECAY_STEPS,
    end_learning_rate=END_EPSILON
    )

agent = dqn_agent.DdqnAgent(
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

replay_observer = [replay_buffer.add_batch]

dataset = replay_buffer.as_dataset(
    sample_batch_size=BUFFER_BATCH_SIZE,
    num_steps=N_STEP_UPDATE + 1,
    num_parallel_calls=3).prefetch(3)

dataset_iterator = iter(dataset)

agent.initialize()

agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    env=train_env,
    policy=agent.collect_policy,
    observers = replay_observer,
    num_steps=COLLECTION_STEPS
    )

num_episodes_metric = tf_metrics.NumberOfEpisodes()
num_steps_metric = tf_metrics.EnvironmentSteps()
avg_return_metric = tf_metrics.AverageReturnMetric()
avg_episode_len_metric = tf_metrics.AverageEpisodeLengthMetric()

eval_metrics = [
            # num_episodes_metric,
            # num_steps_metric,
            avg_return_metric,
            avg_episode_len_metric
]

# def compute_avg_return(environment, policy, num_episodes=10):

#   total_return = 0.0
#   for _ in range(num_episodes):

#
#     time_step = environment.reset()
#     episode_return = 0.0
#     num_steps = 0

#     while not time_step.is_last():
#       num_steps += 1
#       action_step = policy.action(time_step)
#       time_step = environment.step(action_step.action)
#       episode_return += time_step.reward
#       print(num_steps)
#     total_return += episode_return

#   avg_return = total_return / num_episodes
#   return avg_return.numpy()[0]

eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env=eval_env,
    policy=agent.policy,
    observers = eval_metrics,
    num_episodes = NUM_EVAL_EPISODES
    )


train_env.reset()
eval_env.reset()

final_time_step, _ = collect_driver.run()

# Initial buffer fill - Use random policy instead?
for i in range(max(int(BUFFER_MAX_LEN/COLLECTION_STEPS), 1)):
    final_time_step, _ = collect_driver.run(final_time_step)

eval_driver.run()

avg_episode_len = avg_episode_len_metric.result().numpy()
avg_return = avg_return_metric.result().numpy()

returns = [avg_return]
episode_lengths = [avg_episode_len]

for metric in eval_metrics:
    metric.reset()

print(f"Average episode length: {avg_episode_len}")
print(f"Average return: {avg_return}")

# Train loop
for _ in range(NUM_TRAINING_ITERATIONS):

    final_time_step, _ = collect_driver.run(final_time_step)
    experience, unused_info = next(dataset_iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % LOG_INTERVAL == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % EVAL_INTERVAL == 0:
        eval_driver.run()
        avg_episode_len = avg_episode_len_metric.result().numpy()
        avg_return = avg_return_metric.result().numpy()
        print(f'Average Return: {avg_return}, '
              + f'Average episode length: {avg_episode_len}')
        returns.append(avg_return)
        episode_lengths.append(avg_episode_len)

        for metric in eval_metrics:
            metric.reset()
