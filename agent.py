#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for creation and training of agents

WARNING: This will run for days. Literally.
My pretty decent laptop takes about 80 hours to finish this.
"""

import os
import pickle

# import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf

# from tf_agents.environments import py_environment
# from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
# from tf_agents.environments import utils
# from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
# from tf_agents.trajectories import time_step as ts

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
# from tf_agents.environments import tf_py_environment
# from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from env import PyEnv2048#, PyEnv2048FlatObservations
from env import PyEnv2048NoBadActions

"""HYPERPARAMETERS"""
NAME = "Run 24" # Name of agent, used for directory and file names

FC_LAYER_PARAMS = (64, 32) # Number and size of hidden dense layers
MAX_DURATION = 500 # Maximum duration of an episode

LEARNING_RATE = 1e-6 # Learning rate for optimizer

DISCOUNT_FACTOR = 0.95 # Discount factor for future rewards (gamma)

ACTIVATION_FN = tf.keras.activations.relu # Activation function
# Optimizer
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
LOSS_FN = common.element_wise_squared_loss # Loss function

BUFFER_MAX_LEN = 500 # Max length of replay buffer
# Size of experience batch passed to the agent each training iteration
BUFFER_BATCH_SIZE = 64
N_STEP_UPDATE = 3 # Number of consecutive transitions
# to pass to the agent at a time during training

# Number of experience steps to collect each training iteration,
# a higher value replaces the whole buffer faster.
COLLECTION_STEPS = 2
NUM_EVAL_EPISODES = 10 # Number of episodes for evaluation
NUM_TRAINING_ITERATIONS = 10000000 # Number of iterations to train for

# Initial epsilon (chance for collection policy to pick random move)
INITIAL_EPSILON = 0.99
END_EPSILON = 0.01 # End epsilon
EPSILON_DECAY_STEPS = 1000000 # How many steps the epsilon should decay over

# Whether to map bad moves to the next good move
USE_BAD_MOVE_MAPPING = True
# Punishment for moves that don't change the state of the game
# (used if USE_BAD_MOVE_MAPPING is set to false)
PUNISHMENT_FOR_BAD_ACTIONS = 16
REWARD_MULTIPLIER = 1 # Multiplier for positive rewards

LOG_INTERVAL = 2000 # How often to print progress to console
EVAL_INTERVAL = 10000 # How often to evaluate the agent's performence

SAVE_DIR = ".." # Where to save checkpoints, policies and stats

# Creates environments for training and evaluation
# Uses wrapper to limit number of moves
# Both environments must be of same type, but can have different
# parameters.
if USE_BAD_MOVE_MAPPING:
    train_py_env = PyEnv2048NoBadActions(REWARD_MULTIPLIER)
else:
    train_py_env = PyEnv2048(PUNISHMENT_FOR_BAD_ACTIONS, REWARD_MULTIPLIER)
train_py_env = wrappers.TimeLimit(
    train_py_env,
    duration=MAX_DURATION
    )
eval_py_env = wrappers.TimeLimit(PyEnv2048(0, 1), duration=MAX_DURATION)

# Converts Py environments to TF environments
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Creates a tensor to count the number of training iterations
train_step_counter = tf.Variable(0)

# Initializes the neural network
q_net = q_network.QNetwork(
        train_env.observation_spec(), # Passes observation spec,
        train_env.action_spec(), # and action spec of environment.
        fc_layer_params=FC_LAYER_PARAMS,
        activation_fn=ACTIVATION_FN)

# Creates a function to handle epsilon decay
epsilon = tf.compat.v1.train.polynomial_decay(
    learning_rate=INITIAL_EPSILON,
    global_step=train_step_counter,
    decay_steps=EPSILON_DECAY_STEPS,
    end_learning_rate=END_EPSILON
    )

# Initializes an agent implementing the DDQN algorithm
agent = dqn_agent.DdqnAgent(
    time_step_spec=train_env.time_step_spec(), # Passes TimeStep spec,
    action_spec=train_env.action_spec(), # and action spec of environment.
    n_step_update=N_STEP_UPDATE,
    q_network=q_net,
    optimizer=OPTIMIZER,
    epsilon_greedy=epsilon,
    td_errors_loss_fn=LOSS_FN,
    gamma=DISCOUNT_FACTOR,
    train_step_counter=train_step_counter
    )

# Initializes replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec, # Passes agent's data spec
    batch_size=train_env.batch_size,
    max_length=BUFFER_MAX_LEN,
    )

# Puts the buffers "add_batch" function in a list to pass as an observer
# to the training driver later
replay_observer = [replay_buffer.add_batch]

# Creates a dataset from the buffer
dataset = replay_buffer.as_dataset(
    sample_batch_size=BUFFER_BATCH_SIZE,
    num_steps=N_STEP_UPDATE + 1,
    num_parallel_calls=3).prefetch(3)

# And an interator from that dataset
dataset_iterator = iter(dataset)

# Initializes the agent
agent.initialize()

# Wraps agent.train in a graph for optimization, can be skipped
agent.train = common.function(agent.train)

# Sets agent's training steps counter to 0
agent.train_step_counter.assign(0)

# Initializes collection driver
collect_driver = dynamic_step_driver.DynamicStepDriver(
    env=train_env,
    policy=agent.collect_policy,
    observers=replay_observer, # Passes the replay buffer observer
    num_steps=COLLECTION_STEPS
    )

# Initializes driver employing random policy
random_policy_driver = dynamic_step_driver.DynamicStepDriver(
    env=train_env,
    policy=random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec()
        ),
    observers=replay_observer,
    num_steps=COLLECTION_STEPS
    )

# Initializes metrics
num_episodes_metric = tf_metrics.NumberOfEpisodes()
num_steps_metric = tf_metrics.EnvironmentSteps()
avg_return_metric = tf_metrics.AverageReturnMetric()
avg_episode_len_metric = tf_metrics.AverageEpisodeLengthMetric()

# Puts them in a list to pass as observers to eval driver
eval_metrics = [
            # num_episodes_metric, # Number of episodes completed
            # num_steps_metric, # Number of steps completed
            avg_return_metric, # Average return (cumulative episode reward)
            avg_episode_len_metric # Average episode length
]

# Initalizes policy saver, to periodically save the agent's policy
policy_saver = PolicySaver(agent.policy)

# Initializes evaluation driver
eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env=eval_env,
    policy=agent.policy,
    observers = eval_metrics,
    num_episodes = NUM_EVAL_EPISODES
    )

"""
Function similar to running the eval driver with the average return
metric, sometimes useful for debugging purposes.
"""
# def compute_avg_return(environment, policy, num_episodes=10):

#   total_return = 0.0
#   for _ in range(num_episodes):


#     time_step = environment.reset()
#     episode_return = 0.0
#     num_steps = 0

#     while not time_step.is_last():
#       num_steps += 1
#       action_step = policy.action(time_step)
#       time_step = environment.step(action_step.action)
#       episode_return += time_step.reward
#     total_return += episode_return

#   avg_return = total_return / num_episodes
#   return avg_return.numpy()[0]

# Resets both environments
train_env.reset()
eval_env.reset()

# Runs the collection driver once to get a time step
final_time_step, _ = collect_driver.run()

# Initial buffer fill using random policy
for i in range(max(int(BUFFER_MAX_LEN/COLLECTION_STEPS), 1)):
    # Can alternatively be run with the collection policy like so:
    # final_time_step, _ = collect_driver.run(final_time_step)
    final_time_step, _ = random_policy_driver.run(final_time_step)

# Runs eval driver once to start it, and get inital performence
eval_driver.run()

# Gets average episode length and average return from metrics
avg_episode_len = avg_episode_len_metric.result().numpy()
avg_return = avg_return_metric.result().numpy()

# Puts them in lists, and initalizes list for losses
returns = [avg_return]
episode_lengths = [avg_episode_len]
losses = []

# Resets all metrics
for metric in eval_metrics:
    metric.reset()

# print(f"Average episode length: {avg_episode_len}")
# print(f"Average return: {avg_return}")

# Creates checkpointer, which periodically creates a backup of these objects,
# and restores them from the latest backup if one is available
checkpointer = common.Checkpointer(
    ckpt_dir=os.path.join(SAVE_DIR, NAME + " data", "checkpoints"),
    max_to_keep=20,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=agent.train_step_counter,
    network=q_net
    )

# Main training loop
for _ in range(NUM_TRAINING_ITERATIONS):

    # Runs collect driver
    final_time_step, _ = collect_driver.run(final_time_step)

    # Gets experiance from buffer
    experience, unused_info = next(dataset_iterator)

    # Train the agent and get the loss
    train_loss = agent.train(experience).loss

    # Save the loss to the losses list
    losses.append(train_loss.numpy())

    # Gets the number of training steps completed
    step = agent.train_step_counter.numpy()

    # Prints progress to console
    if step % LOG_INTERVAL == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    # Evaluates agent performence
    if step % EVAL_INTERVAL == 0:

        # Runs evaluation driver
        eval_driver.run()

        # Gets average episode length and average return from metrics
        avg_episode_len = avg_episode_len_metric.result().numpy()
        avg_return = avg_return_metric.result().numpy()

        # Prints eval results to console
        print(f'Average Return: {avg_return}, '
              + f'Average episode length: {avg_episode_len}')

        # Appends returns and episode lengths to their repsecitve lists
        returns.append(avg_return)
        episode_lengths.append(avg_episode_len)

        # Saves agent policy
        policy_saver.save(
            os.path.join(
            SAVE_DIR, NAME + " data", "policy saves",
            NAME + " policy @ " + str(step)
            )
        )

        # Runs checkpointer to make a backup of the agent, network etc.
        checkpointer.save(step)

        # Saves the lists of statistics as a pickled dictionary
        with open(
                os.path.join(SAVE_DIR, NAME + " data", NAME + " stats.pkl"),
                "wb",
                ) as file:
            pickle.dump(
                {"Returns": returns,
                 "Lengths": episode_lengths,
                 "Losses": losses},
                file
                )

        # Resets all metrics
        for metric in eval_metrics:
            metric.reset()
