#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pg_implementation
import env

from tf_agents.policies import random_py_policy
from tf_agents.policies import random_tf_policy
from tf_agents.environments import tf_py_environment

game = env.PyEnv2048(0)
game = tf_py_environment.TFPyEnvironment(game)


class RandomBot:
    def __init__(self, env):
        self.policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                env.action_spec())
    def get_action(self, ts):
        return self.policy.action(ts).action

pg_implementation.main(game, RandomBot(game))