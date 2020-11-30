#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

class PyEnv2048(py_environment.PyEnvironment):

    def __init__(self):

        # Specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.uint8, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,4), dtype=np.uint64, minimum=0, name='observation')

        # Grid with two initial values
        self._state = np.zeros(shape=(4,4), dtype=np.uint64)
        a, b = random.sample([(x,y) for x in range(4) for y in range(4)], 2)
        self._state[a[0]][a[1]] = 2
        self._state[b[0]][b[1]] = 2

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        # Grid with two initial values
        self._state = np.zeros(shape=(4,4), dtype=np.uint64)
        a, b = random.sample([(x,y) for x in range(4) for y in range(4)], 2)
        self._state[a[0]][a[1]] = 2
        self._state[b[0]][b[1]] = 2

        self._episode_ended = False

        return ts.restart(self._state)

    def __gameover(self):

        # Return false if there are empty tiles left
        if np.argwhere(self._state == 0).any():
            return False

        # checks if any tiles can be merged
        for y in range(4):
            for x in range(4):

                # Only checks right and up, since if it can't be merged
                # up it can't be merged down either, and same with
                # right and left, and we are checking all tiles.

                # Excepts IndexError, you can't match out of bounds
                # anyway.

                #??? Maybe refeactor to remove dublicate try clauses?
                try:
                    # Checks up
                    if self._state[y][x] == self._state[y+1][x]:
                        # If can merge, game is not over
                        return False
                except IndexError:
                    pass

                try:
                    # Checks right
                    if self._state[y][x] == self._state[y][x+1]:
                        return False
                except IndexError:
                    pass

        # If this point has been reached, all tiles have been checked for
        # merges, and no possible merges have been found.
        return True


    def __new_tile(self):

        # If there are no empty tiles, return false
        if not (empty := np.argwhere(self._state == 0)).any():
            return False

        # 90% chance that the new tile is 2, otherwise 4
        if random.random() < 0.9:
            new = 2
        else:
            new = 4

        # Set a random empty tile to the new value
        y, x = random.choice(empty)
        self._state[y][x] = new

        # bool((x,y)) == True
        return (x, y)

    def _step(self, action):

        # Reset if game over
        if self._episode_ended:
            return self.reset()

        #TODO

        #DEFINE "reward" here:
        if action == 0:
            # move up
            pass
        elif action == 1:
            # move right
            pass
        elif action == 2:
            # move down
            pass
        elif action == 3:
            # move left
            pass

        # Check whether game has ended, or if a set number of moves was made
        if self._episode_ended or self.__gameover():
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)

