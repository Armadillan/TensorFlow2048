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

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

num_iterations = 10000

initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_capacity = 1000

fc_layer_params = (100,100)

batch_size = 128
learning_rate = 1e-2
log_interval = 200

num_eval_episodes = 10
eval_interval = 1000

# Sets maximum number of steps to 500
# train_py_env = wrappers.TimeLimit(PyEnv2048(), duration=500)
# eval_py_env = wrappers.TimeLimit(PyEnv2048(), duration=500)

# Turns py environments into tf environments
train_env = tf_py_environment.TFPyEnvironment(PyEnv2048())
eval_env = tf_py_environment.TFPyEnvironment(PyEnv2048())

q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

def get_legal_moves(state):

        legal = [0,0,0,0]

        # test up
        for y in range(4):
            for x in range(4):
                if (tile_value := state[y][x]) != 0:

                    new_y = y

                    # Moves the tile up as far as it can go
                    while new_y > 0 and state[new_y-1][x] == 0:
                        new_y -= 1

                    # Checks if the tile can be merged, and merges
                    if (new_y > 0 \
                        and tile_value == state[new_y-1][x]) \
                        or new_y != y:
                            legal[0] = 1
                            break
            if legal[0]:
                break

        #test right
        for y in range(4):
            for x in range(3, -1, -1):

                if (tile_value := state[y][x]) != 0:
                    new_x = x

                    while new_x < 3 and state[y][new_x+1] == 0:
                        new_x += 1

                    if (new_x < 3 \
                        and tile_value == state[y][new_x + 1]) \
                        or new_x != x:

                            legal[1] = 1
                            break
            if legal[1]:
                break

        #test down
        for y in range(3, -1, -1):
            for x in range(4):
                if (tile_value := state[y][x]) != 0:
                    new_y = y

                    while new_y < 3 and state[new_y+1][x] == 0:
                        new_y += 1

                    if (new_y < 3 \
                        and tile_value == state[new_y+1][x]) \
                        or new_y != y:

                            legal[2] = 1
                            break
            if legal[2]:
                break

        # test left
        for y in range(4):
            for x in range(4):
                if (tile_value := state[y][x]) != 0:
                    new_x = x

                    while new_x > 0 and state[y][new_x-1] == 0:
                        new_x -= 1

                    if (new_x > 0 \
                        and tile_value == state[y][new_x-1]) \
                        or new_x != x:

                            legal[3] = 1
            if legal[3]:
                break

        return tf.convert_to_tensor(legal, dtype=np.int32)

train_step_counter = tf.compat.v2.Variable(0)

def splitter(observation):
    if isinstance(observation, tf.TensorSpec):
        return (observation, tf.TensorSpec(
            shape=(4,), dtype=tf.int32)
                )
    state = observation.numpy()[0]
    return (observation, get_legal_moves(state))


tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        n_step_update=4,
        q_network=q_net,
        optimizer=optimizer,
        observation_and_action_constraint_splitter=None,
        td_errors_loss_fn = common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

replay_observer = [replay_buffer.add_batch]

train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
]

def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
    num_steps=5).prefetch(3)

driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            collect_policy,
            observers=replay_observer + train_metrics,
    num_steps=1)

iterator = iter(dataset)


policy = tf_agent.policy

time_step = train_env.reset()
actions_and_steps = [time_step]


tf_agent.train = common.function(tf_agent.train)
tf_agent.train_step_counter.assign(0)

avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

eval_env.reset()
train_env.reset()

final_time_step, policy_state = driver.run()


for i in range(1000):
    final_time_step, _ = driver.run(final_time_step, policy_state)

episode_len = []
step_len = []

returns_x = []


for i in range(num_iterations):
    final_time_step, _ = driver.run(final_time_step, policy_state)
    #for _ in range(1):
    #    collect_step(train_env, tf_agent.collect_policy)

    experience, _ = next(iterator)
    train_loss = tf_agent.train(experience=experience)
    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
        episode_len.append(train_metrics[3].result().numpy())
        step_len.append(step)

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('Average episode length: {}'.format(train_metrics[3].result().numpy()))
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
        returns_x.append(step)

fig, ax = plt.subplots()

ax.plot(step_len, episode_len)
ax.set_xlabel('Episodes')
ax.set_ylabel('Average Episode Length (Steps)')

fig, ax = plt.subplots()

ax.plot([0]+returns_x, returns)
ax.set_xlabel('Episodes')
ax.set_ylabel('Average returns')
