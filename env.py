#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import tensorflow as tf
import numpy as np

# unused imports are sometimes used in the console while developing
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

class PyEnv2048(py_environment.PyEnvironment):

    MAX_MOVES = 1000

    def __init__(self):

        # Specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int8, minimum=0, maximum=3, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,4), dtype=np.uint64, minimum=0, name='observation')

        # Grid with two initial values
        self._state = np.zeros(shape=(4,4), dtype=np.uint64)
        a, b = random.sample([(x,y) for x in range(4) for y in range(4)], 2)
        self._state[a[0]][a[1]] = 2
        self._state[b[0]][b[1]] = 2

        self._episode_ended = False

        self._moves = 0

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

        self._moves = 0

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

                # Excepts IndexError, you can't match out of bounds anyway.

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

        # Reset if episode over
        # This code is unreachable, but it is in the tutorial for some reason,
        # so I put it here as well, maybe it has some use I don't know of?
        if self._episode_ended:
            return self.reset()

        self._moves += 1
        if self._moves >= self.MAX_MOVES:
            self._episode_ended = True

        #??? Somehow refactor to remove a lot of dublicate code?

        # list for tiles already merged this move
        merged = []
        reward = 0

        moved = False

        # move up
        if action == 0:

            # Starts at the top
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
                                # Sets moved to True
                                moved = True

                        # If it can not be merged, just moves it
                        elif new_y != y:
                            self._state[y][x] = 0
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

        # If moved, add new tile
        if moved:
            self.__new_tile()

        discount = 1
        #!!! Possible to add discount as to not discourage moves that do not
        # give any points.
        # if reward == 0 and moved:
        #     discount = 1.1

        # Check whether game has ended, or if a set number of moves was made
        if self._episode_ended or self.__gameover():
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=discount)

if __name__ == "__main__":
    try:
        environment = PyEnv2048()
        utils.validate_py_environment(environment, episodes=5)
    except:
        raise
    else:
        print("No exceptions :)")
