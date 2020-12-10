#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pg_implementation
import env
import tensorflow as tf


from tf_agents.policies import random_tf_policy
from tf_agents.environments import tf_py_environment
# from tf_agents.policies.policy_saver import PolicySaver

game = tf_py_environment.TFPyEnvironment(env.PyEnv2048())


# policy = random_tf_policy.RandomTFPolicy(game.time_step_spec(),
#                                                 game.action_spec())
policy = tf.compat.v2.saved_model.load('..\\Run 18 policy saves\\Run 18 policy @ 1780000')

pg_implementation.Game(game, policy, 0.1).start()