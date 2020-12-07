#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pg_implementation
import env


from tf_agents.policies import random_tf_policy
from tf_agents.environments import tf_py_environment

game = tf_py_environment.TFPyEnvironment(env.PyEnv2048(0))


policy = random_tf_policy.RandomTFPolicy(game.time_step_spec(),
                                                game.action_spec())

pg_implementation.main(game, policy, 0.1)