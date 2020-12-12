#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes for python environments suitable for reinforcement learning with
the tf_agents library.
"""

import random

# import tensorflow as tf
import numpy as np

# unused imports are sometimes used in the console while developing
from tf_agents.environments import py_environment
# from tf_agents.environments import tf_environment
# from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
# from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

class PyEnv2048(py_environment.PyEnvironment):
    """

    2048 as a tf_agents.environments.py_environment.PyEnvironment object
    Handles all the game logic.
    Can be turned into a TensorFlow environment using the TFPyEnvironment
    wrapper.

    Implements variable negative rewards for moves
    that don't change the state of the game,
    and an adjustable reward multiplier.
    Setting these to 0 and 1, respectively, results in behavior
    identical to the original game.
    The reward multiplier is applied only to positive rewards,
    not punishments.

    """
    def __init__(self, neg_reward=0, reward_multiplier=1):

        """
        Initializes the object, starts the game
        """

        # Specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,4), dtype=np.int64, minimum=0, name='observation')

        # Initializes the game board, a numpy.ndarray
        self._state = np.zeros(shape=(4,4), dtype=np.int64)
        # with two starting twos in random locations
        a, b = random.sample(((x,y) for x in range(4) for y in range(4)), 2)
        self._state[a[0]][a[1]] = 2
        self._state[b[0]][b[1]] = 2

        self._episode_ended = False # Whether the game is over or not

        self._neg_reward = neg_reward
        self._reward_multiplier = reward_multiplier

        self._moves = 0 # Counter for the number of moves made

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        """

        Resets the environment, restarts the game.

        """
        # Grid with two initial values
        self._state = np.zeros(shape=(4,4), dtype=np.int64)
        a, b = random.sample([(x,y) for x in range(4) for y in range(4)], 2)
        self._state[a[0]][a[1]] = 2
        self._state[b[0]][b[1]] = 2

        self._episode_ended = False

        # Returns "restart" TimeStep with the state of the game
        return ts.restart(self._state)

    def __gameover(self):
        """

        Checks if the game is over

        """
        # Return false if there are empty tiles left
        if not self._state.all():
            return False

        # Checks if any tiles can be merged
        for y in range(4):
            for x in range(4):

                # Only checks right and down, since if it can't be merged
                # down it can't be merged up either, and same with
                # right and left. All tiles are checked anyway.

                # Excepts IndexError, you can't match out of bounds anyway.

                try:
                    # Checks down
                    if self._state[y][x] == self._state[y+1][x]:
                        # If merge is possible, game is not over
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
        self._episode_ended = True
        return True



    def __new_tile(self):
        """

        Creates a new tile on the board.
        The returns are not used for anything right now, but they could be,
        so I'm leaving them in.

        """

        # If there are no empty tiles, return False
        # Also creates list of all indices of empty tiles (empty)
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

        # bool((x,y)) == True, returns "True" if a new tile was created.
        return (x, y)

    def _step(self, action):
        """

        Expects action in (0, 1, 2, 3)
        Accepts tf_agents.trajectories.policy_step.PolicyStep.action
        from both TF and Py policies
        """

        # Reset if episode is over
        # This code is unreachable, but it is in the tutorial for some reason,
        # so I put it here as well, maybe it has some use I don't know of?
        if self._episode_ended:
            return self.reset()

        # List for tiles already merged this move
        # tiles should not be merged twice
        merged = []

        reward = 0 # Cumulative reward for all merges

        moved = False # Whether the board changed this move


        # Performs move based on action:

        # move up
        if action == 0:

            # Starts at the top (0,0), moving down and right,
            # Loops through all tiles
            for y in range(4):
                for x in range(4):

                    # only moves non-zero tiles
                    if (tile_value := self._state[y][x]) != 0:

                        new_y = y

                        # Moves the tile up as far as it can go
                        while new_y > 0 and self._state[new_y-1][x] == 0:
                            new_y -= 1

                        # Checks if the tile can be merged, and merges
                        if new_y > 0 \
                            and tile_value == self._state[new_y-1][x] \
                            and (new_y - 1, x) not in merged:

                            # Sets the old location to 0
                            self._state[y][x] = 0
                            # Doubles the new location (merge)
                            self._state[new_y-1][x] *= 2
                            # Appends tile to merged list
                            merged.append((new_y-1, x))
                            # Adds reward
                            reward += tile_value * 2

                            moved = True

                        # If it can not be merged, just moves it
                        elif new_y != y:
                            # Sets old location to 0
                            self._state[y][x] = 0
                            # Sets new location to the value of the tile
                            self._state[new_y][x] = tile_value
                            moved = True


        # move right
        elif action == 1:

            for y in range(4):
                # Start at the far right
                for x in range(3, -1, -1):

                    if (tile_value := self._state[y][x]) != 0:
                        new_x = x

                        while new_x < 3 and self._state[y][new_x+1] == 0:
                            new_x += 1

                        if new_x < 3 \
                            and tile_value == self._state[y][new_x + 1] \
                            and (y, new_x + 1) not in merged:

                            self._state[y][x] = 0
                            self._state[y][new_x+1] *= 2
                            merged.append((y, new_x+1))
                            reward += tile_value * 2
                            moved = True

                        elif new_x != x:
                            self._state[y][x] = 0
                            self._state[y][new_x] = tile_value
                            moved = True

        # move down
        elif action == 2:

            # start at the bottom
            for y in range(3, -1, -1):
                for x in range(4):
                    if (tile_value := self._state[y][x]) != 0:
                        new_y = y

                        while new_y < 3 and self._state[new_y+1][x] == 0:
                            new_y += 1

                        if new_y < 3 \
                            and tile_value == self._state[new_y+1][x] \
                            and (new_y + 1, x) not in merged:

                            self._state[y][x] = 0
                            self._state[new_y+1][x] *= 2
                            merged.append((new_y+1, x))
                            reward += tile_value * 2
                            moved = True

                        elif new_y != y:
                            self._state[y][x] = 0
                            self._state[new_y][x] = tile_value
                            moved = True

        # move left
        elif action == 3:

            for y in range(4):
                # start at the far left
                for x in range(4):
                    if (tile_value := self._state[y][x]) != 0:
                        new_x = x

                        while new_x > 0 and self._state[y][new_x-1] == 0:
                            new_x -= 1

                        if new_x > 0 \
                            and tile_value == self._state[y][new_x-1] \
                            and (y, new_x - 1) not in merged:

                            self._state[y][x] = 0
                            self._state[y][new_x - 1] *= 2
                            merged.append((y, new_x - 1))
                            reward += tile_value * 2
                            moved = True

                        elif new_x != x:
                            self._state[y][x] = 0
                            self._state[y][new_x] = tile_value
                            moved = True

        # If moved, add new tile and applies reward multiplier
        if moved:
            self.__new_tile()
            reward *= self._reward_multiplier

        else:
            # If not moved, applies punishment
            reward = - self._neg_reward

        # Check whether game has ended
        if self._episode_ended or self.__gameover():
            # Returns "termination" TimeStep with current state and reward
            return ts.termination(self._state, reward)

        # If the game has not ended, returns "transiton" TimeStep
        return ts.transition(self._state, reward)

class PyEnv2048FlatObservations(PyEnv2048):
    """

    The same as PyEnv2048 but the observation has
    shape (16,) instead of (4,4)

    """

    def __init__(self, neg_reward=0, reward_multiplier=1):
        """
        Calls __init__ from PyEnv2048 and redefines observation spec
        to reflect the new shape.
        """
        super().__init__(neg_reward, reward_multiplier)
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(16,), dtype=np.int64, minimum=0, name='observation')

    def _step(self, action):
        """

        Gets the TimeStep from PyEnv2048._step and then returns another
        with the same content, but the observation array is flattened

        """
        time_step = super()._step(action)
        return ts.TimeStep(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=time_step.observation.flatten())

    def _reset(self):
        """

        Gets the TimeStep from PyEnv2048._reset and then returns another
        with the same content, but the observation array is flattened

        """
        time_step = super()._reset()
        return ts.TimeStep(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=time_step.observation.flatten())

if __name__ == "__main__":

    # Here are some basic tests
    try:
        environment = PyEnv2048()
        utils.validate_py_environment(environment, episodes=5)
    except:
        raise
    else:
        print("No exceptions :)")

    try:
        environment = PyEnv2048FlatObservations()
        utils.validate_py_environment(environment, episodes=5)
    except:
        raise
    else:
        print("No exceptions :)")
