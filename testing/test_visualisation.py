#!/usr/bin/env python
# -*- coding: utf-8 -*-

import test_pg_implementation
import env
import tensorflow as tf


from tf_agents.policies import random_tf_policy
from tf_agents.environments import tf_py_environment
from tf_agents.policies.policy_saver import PolicySaver

game = tf_py_environment.TFPyEnvironment(env.PyEnv2048(0))


policy = random_tf_policy.RandomTFPolicy(game.time_step_spec(),
                                                game.action_spec())
# policy = tf.compat.v2.saved_model.load('..\\Run 15 policy saves\\Run 15 policy @ 1000000')

test_pg_implementation.main(game, policy)