#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_agents.policies import random_tf_policy
from tf_agents.policies import random_py_policy
from tf_agents.environments import tf_py_environment

import pg_implementation
import env

"""Create TF environment like so:"""
game = tf_py_environment.TFPyEnvironment(env.PyEnv2048())

"""or Py environment like so:"""
# game = env.PyEnv2048()


"""Random TF policy like so: (Requires TF environment)"""
policy = random_tf_policy.RandomTFPolicy(
    game.time_step_spec(), game.action_spec()
    )

"""or random Py policy like so: (Requires Py environment)"""
# policy = random_py_policy.RandomPyPolicy(
#     game.time_step_spec(), game.action_spec()
#     )

"""or load a saved policy like so: (requires compatible environment)"""
policy = tf.compat.v2.saved_model.load(
    '..\\Run 17 policy saves\\Run 17 policy @ 3900000'
    )

"""Start the game: (initializes object and calls it's start() method)"""
pg_implementation.Game(game, policy, 0.1).start()
