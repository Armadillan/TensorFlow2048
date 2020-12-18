#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class for a visual interface for the 2048 environment using pygame.
Playable by both humans and robots.
"""

import time
import os

import pygame
import numpy as np


class Game:
    """
    Class for a visual interface for the 2048 environment using pygame.
    Playable by both humans and robots.

    Use arrow keys or WASD to control.
    r restarts the game
    b turns bot on or off

    """

    # Whether the game always stays square with
    # background color on the sides, or fills the
    # whole window.
    # Setting this to False currently results in font weirdness when
    # the window is much taller than it's wide. It's fixable but I
    # haven't had time to fix it yet.
    SQUARE = True

    BACKGROUND_COLOR = ("#bbada0") # Background color (duh?)

    LIGHT_TEXT_COLOR = pygame.Color("#f9f6f2") # Lighter color of digits
    DARK_TEXT_COLOR = ("#776e65") # Darker color of digits
    # Which tiles to use the lighter font for
    # All other tiles will use the darker font
    VALUES_WITH_DARK_TEXT = (2, 4)

    # Dictionary mapping tiles to their color
    TILE_COLOR = {
        0: ("#cdc1b4"),
        2: pygame.Color("#eee4da"),
        4: pygame.Color("#eee1c9"),
        8: pygame.Color("#f3b27a"),
        16: pygame.Color("#f69664"),
        32: pygame.Color("#f77c5f"),
        64: pygame.Color("#f75f3b"),
        128: pygame.Color("#edd073"),
        256: pygame.Color("#edcc62"),
        512: pygame.Color("#edc950"),
        1024: pygame.Color("#edc53f"),
        2048: pygame.Color("#edc22e"),
        "BIG": pygame.Color("#3c3a33") # Tiles bigger than 2048
        }

    def __init__(self, env, bot=None, bot_delay=0.1):
        """
        Iniitalizes the object.

        Parameters
        ----------
        env : PyEnvironment or TFPyEnvironment
            An environment from env.py,
            possibly wrapped using TFPyEnvironment
        bot : TFPolicy or PyPolicy or TFPyPolicy or equivalent, optional
            A robot to play the game.
            Must have an action(TimeStep) method that
            returns and action compatible with the environment.
            The default is None.
        bot_delay : float, optional
            The delay in seconds between bot moves. The default is 0.1.

        Returns
        -------
        None.

        """

        # Adds text to caption if there is no bot
        if bot is None:
            self.caption_end = " " * 10 + "NO BOT ATTACHED"
        else:
            self.caption_end = ""

        self.env = env
        self.bot = bot
        self.bot_delay = bot_delay

        # Initial size for the window
        self.w = 600
        self.h = 600

        # Initializes pygame
        pygame.init()

        # Initializes fonts
        self.initialize_fonts()

    def initialize_fonts(self):
        """

        Initializes fonts based on current screen size
        Must be called every time screen size changes

        """
        # Using the original font from 2048
        self.tile_font_5 = pygame.font.Font(
            os.path.join("assets", "ClearSans-Bold.ttf"),
            int(self.h * 6/29 * 7/12)
            )
        self.tile_font_4 = pygame.font.Font(
            os.path.join("assets", "ClearSans-Bold.ttf"),
            int(self.h * 6/29 * 6/12)
            )
        self.tile_font_3 = pygame.font.Font(
            os.path.join("assets", "ClearSans-Bold.ttf"),
            int(self.h * 6/29 * 5/12)
            )
        self.tile_font_2 = pygame.font.Font(
            os.path.join("assets", "ClearSans-Bold.ttf"),
            int(self.h * 6/29 * 4/12)
            )
        self.tile_font_1 = pygame.font.Font(
            os.path.join("assets", "ClearSans-Bold.ttf"),
            int(self.h * 6/29 * 3/12)
            )

        # If that is not possible for some reason,
        # Arial can be used:

        # self.tile_font_5 = pygame.font.SysFont(
        #     "Arial", int(self.h * 5/29)
        #     )
        # self.tile_font_4 = pygame.font.SysFont(
        #     "Arial", int(self.h * 4/29)
        #     )
        # self.tile_font_3 = pygame.font.SysFont(
        #     "Arial", int(self.h * 3/29)
        #     )
        # self.tile_font_2 = pygame.font.SysFont(
        #     "Arial", int(self.h * 2/29)
        #     )
        # self.tile_font_1 = pygame.font.SysFont(
        #     "Arial", int(self.h * 1/29)
        #     )

        self.gameover_font_1 = pygame.font.SysFont(
            "Arial", int(self.h * (1/10)), bold=True
            )
        self.gameover_size_1 = self.gameover_font_1.size("GAME OVER")
        self.gameover_text_1 = self.gameover_font_1.render(
            "GAME OVER", True, (20,20,20)
            )

        self.gameover_font_2 = pygame.font.SysFont(
            "Arial", int(self.h * (1/20),), bold=True
            )
        self.gameover_size_2 = self.gameover_font_2.size(
            "Press \"r\" to restart"
            )
        self.gameover_text_2 = self.gameover_font_2.render(
            "Press \"r\" to restart", True, (20,20,20)
                )

    def tile(self, x, y, n):
        """
        Used to render tiles

        Parameters
        ----------
        x : int
            x coordinate of tile.
        y : int
            y coordinate of tile.
        n : int
            Value of tile.

        Returns
        -------
        pygame.Rect
            Rectangle making up the tiles background.
        pygame.Surface
            The text (number) on the tile.
        tuple
            (x, y) coordinates of the text on the tile.

        """

        # Coordinates of tile
        rect_x = (x+1) * self.w/29 + x * self.w * (6/29)
        rect_y = (y+1) * self.h/29 + y * self.h * (6/29)
        # Rectangle object
        rect = pygame.Rect(
        rect_x, rect_y, self.w * (6/29), self.h * (6/29))

        # Does not render text if the tile is 0
        if not n:
            text_render = pygame.Surface((0,0))
            text_x = 0
            text_y = 0

        else:

            # Get string from int and it's length
            text = str(n)
            l = len(text)

            # Chooses color for text
            if n in self.VALUES_WITH_DARK_TEXT:
                text_color = self.DARK_TEXT_COLOR
            else:
                text_color = self.LIGHT_TEXT_COLOR

            # Chooses font size based on length of text
            if l < 3:
                font = self.tile_font_5
            elif l == 3:
                font = self.tile_font_4
            elif l == 4:
                font = self.tile_font_3
            elif l < 7:
                font = self.tile_font_2
            else:
                font = self.tile_font_1

            # Renders font
            text_render = font.render(text, True, text_color)
            # Gets size of text
            size = font.size(text)

            # Calculates text coordinates
            text_x = (x+1) * self.w/29 \
                + (x+0.5) * self.w * (6/29) - size[0] / 2
            text_y = (y+1) * self.h/29 \
                + (y+0.5) * self.h * (6/29) - size[1] / 2

        return rect, text_render, (text_x, text_y)

    def main(self):
        """

        Starts the game.
        This is the main game loop.

        """

        # Initial status
        playing = True
        gameover = False
        bot_active = False
        score  = 0
        moves = 0

        # Gets initial game state
        ts = self.env.reset()

        # This is for compatibility with different types of environments
        try:
            # TF environments
            board_array = ts.observation.numpy()[0]
        except AttributeError:
            # Py environments
            board_array = ts.observation
        # Environments with flat observations
        board_array = np.reshape(board_array, (4,4))

        # Keeps a counter of score and moves made in the caption
        pygame.display.set_caption(
            "2048" + " " * 10 + "Score: 0   Moves: 0" + self.caption_end
            )

        # Initializes window and a drawing surface
        win = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
        surface = win.copy()

        # Main game loop
        while playing:

            moved = False

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    playing = False
                    break

                if event.type == pygame.VIDEORESIZE:
                    # Handles window resizing

                    self.w, self.h = win.get_size()
                    # Sets width and height equal if SQUARE is True
                    if self.SQUARE:
                        if self.w > self.h:
                            self.w = self.h
                        else:
                            self.h = self.w
                    # Makes new drawing surface
                    surface = win.copy()
                    # Re-initalizes fonts based on new window size
                    self.initialize_fonts()

                if event.type == pygame.KEYDOWN:
                    # Handles user input

                    if event.key == pygame.K_r:
                        #Restarts the game
                        ts = self.env.reset()
                        score = 0
                        moves = 0
                        moved = True
                        gameover = False
                        surface.fill(self.BACKGROUND_COLOR)

                    elif event.key == pygame.K_b:
                        # Turns bot off and on
                        if bot_active:
                            bot_active = False
                            pygame.display.set_caption(
                            "2048" + " " * 10 + f"Score: {int(score)}"
                            + f"   Moves: {moves}"
                            )
                        elif self.bot is not None:
                            bot_active = True

                    elif not gameover:
                        # Handles human player moves

                        if event.key in (pygame.K_UP, pygame.K_w):
                            ts = self.env.step(0)
                            moved = True
                            moves += 1
                        elif event.key in (pygame.K_RIGHT, pygame.K_d):
                            ts = self.env.step(1)
                            moved = True
                            moves += 1
                        elif event.key in (pygame.K_DOWN, pygame.K_s):
                            ts = self.env.step(2)
                            moved = True
                            moves += 1
                        elif event.key in (pygame.K_LEFT, pygame.K_a):
                            ts = self.env.step(3)
                            moved = True
                            moves += 1

            # Breaks loop if game is over
            if not playing:
                break

            # Updates game state if a move has been made by the player
            if moved:
                try:
                    board_array = ts.observation.numpy()[0]
                    score += ts.reward.numpy()
                except AttributeError:
                    board_array = ts.observation
                    score += ts.reward
                board_array = np.reshape(board_array, (4,4))

            # Handles bot movements
            if not gameover:

                if bot_active:

                    old_board = board_array.copy()

                    # Gets action from bot
                    actionstep = self.bot.action(ts)
                    action = actionstep.action

                    ts = self.env.step(action)
                    moves += 1

                    # Updates game state
                    try:
                        board_array = ts.observation.numpy()[0]
                        score += ts.reward.numpy()
                    except AttributeError:
                        board_array = ts.observation
                        score += ts.reward
                    board_array = np.reshape(board_array, (4,4))

                    # Checks if the board has changed
                    if not np.array_equal(old_board, board_array):
                        moved = True
                        # Waits before bot makes another move
                        time.sleep(self.bot_delay)

                    # Adds "- Bot" to caption
                    pygame.display.set_caption(
                        "2048 - Bot" + " " * 10 + f"Score: {int(score)}"
                        + f"   Moves: {moves}"
                        )

                else:
                    # Updates caption without "- Bot"
                    pygame.display.set_caption(
                        "2048" + " " * 10 + f"Score: {int(score)}"
                        + f"   Moves: {moves}"+ self.caption_end
                    )

                # Checks if the game is over
                if ts.is_last():
                    gameover = True

            # Draws all the graphics:
            surface.fill(self.BACKGROUND_COLOR)

            # Draws every tile
            for x in range(4):
                for y in range(4):

                    # Gets the tile "data"
                    n = board_array[y][x]
                    rect, text, text_coords = self.tile(x, y, n)

                    # Gets the color of the tile
                    try:
                        tile_color = self.TILE_COLOR[n]
                    except KeyError:
                        tile_color = self.TILE_COLOR["BIG"]

                    # Draws the background
                    pygame.draw.rect(
                        surface=surface,
                        color=tile_color,
                        rect=rect,
                        border_radius=int((self.w+self.h)/2 * 1/150)
                        )

                    # Blits the text surface to the drawing surface
                    surface.blit(text, text_coords)

            # Displays "gameover screen" if game is over
            if gameover:

                x_1 = self.w / 2 - self.gameover_size_1[0] / 2
                y_1 = self.h / 2 - self.h * 6/80
                x_2 = self.w / 2 - self.gameover_size_2[0] / 2
                y_2 = self.h / 2 + self.h * 1/30
                surface.blit(self.gameover_text_1, (x_1, y_1))
                surface.blit(self.gameover_text_2, (x_2, y_2))

            # Fill window with background color
            win.fill(self.BACKGROUND_COLOR)

            # Blits drawing surface to the middle of the window
            w, h = win.get_size()
            if w > h:
                win.blit(surface, ((win.get_width()-self.w)/2,0))
            else:
                win.blit(surface, (0, (win.get_height()-self.h)/2))

            # Updates display
            pygame.display.update()

        # Quits pygame outside of the main game loop, if the game is over
        pygame.quit()

if __name__ == "__main__":
    # Plays the game without a bot attached :)
    from env import PyEnv2048

    # Creates Game object, passing an environment to the constructor
    game = Game(PyEnv2048())
    # Starts the interface
    game.main()
