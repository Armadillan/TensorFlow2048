#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pg_implementation
import env

from tf_agents.policies import random_py_policy

game = env.PyEnv2048(0)

class RandomBot:
    def __init__(self, env):
        self.policy = random_py_policy.RandomPyPolicy(env.time_step_spec(),
                                                env.action_spec())
    def get_action(self, ts):
        return self.policy.action(ts).action

pg_implementation.main(game, RandomBot(game))