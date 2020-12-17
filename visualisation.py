#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Script using pg_implementation to see a policy in action.

Here you can see how to load a policy that makes random moves, or a policy
from one of the agents I trained.

When the game is running, press b to start the bot.
Mismatching policy and environment (mixing Py and TF) can result in a
variety of errors, so I haven't bothered to catch any of them.
Just don't do it.

"""

import os

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


"""Create a random TF policy like so: (Requires TF environment)"""
# policy = random_tf_policy.RandomTFPolicy(
#     game.time_step_spec(), game.action_spec()
#     )

"""or random Py policy like so: (Requires Py environment)"""
# policy = random_py_policy.RandomPyPolicy(
#     game.time_step_spec(), game.action_spec()
#     )

"""or load a saved policy like so: (requires compatible environment)"""
policy = tf.compat.v2.saved_model.load(
    # This one is a TF policy, and requires a TF environment
    os.path.join("assets", "bad_bot")
    # This is the same bot, but after more training:
    # It is somewhat worse.
    # os.path.join("assets", "bad_bot_2")
    )

"""Start the game: (initializes Game object and calls its main() method)"""
pg_implementation.Game(game, policy, 0.1).main()
