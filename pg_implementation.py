#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import pygame
import numpy as np


class Game:

    BACKGROUND_COLOR = (187, 173, 160)

    SQUARE = True

    LIGHT_TEXT_COLOR = pygame.Color("#f9f6f2")
    DARK_TEXT_COLOR = (119, 110, 101)
    VALUES_WITH_DARK_TEXT = (2, 4)

    TILE_COLOR = {
        0: (205, 193, 180),
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
        "BIG": pygame.Color("#3c3a33")
        }

    def __init__(self, env, bot=None, bot_delay=0.1):

        if bot is None:
            self.caption_end = " " * 10 + "NO BOT ATTACHED"
        else:
            self.caption_end = ""

        self.env = env
        self.bot = bot
        self.bot_delay = bot_delay
        self.w = 600
        self.h = 600

        pygame.init()

        self.initialize()

    def initialize(self):

        self.tile_font_5 = pygame.font.Font(
            "ClearSans-Bold.ttf", int(self.h * 6/29 * 7/12)
            )
        self.tile_font_4 = pygame.font.Font(
            "ClearSans-Bold.ttf", int(self.h * 6/29 * 6/12)
            )
        self.tile_font_3 = pygame.font.Font(
            "ClearSans-Bold.ttf", int(self.h * 6/29 * 5/12)
            )
        self.tile_font_2 = pygame.font.Font(
            "ClearSans-Bold.ttf", int(self.h * 6/29 * 4/12)
            )
        self.tile_font_1 = pygame.font.Font(
            "ClearSans-Bold.ttf", int(self.h * 6/29 * 3/12)
            )

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

        rect_x = (x+1) * self.w/29 + x * self.w * (6/29)
        rect_y = (y+1) * self.h/29 + y * self.h * (6/29)
        rect = pygame.Rect(
        rect_x, rect_y, self.w * (6/29), self.h * (6/29))

        if not n:
            text_render = pygame.Surface((0,0))
            text_x = 0
            text_y = 0

        else:

            text = str(n)
            l = len(text)

            if n in self.VALUES_WITH_DARK_TEXT:
                text_color = self.DARK_TEXT_COLOR
            else:
                text_color = self.LIGHT_TEXT_COLOR

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

            text_render = font.render(text, True, text_color)
            size = font.size(text)

            text_x = (x+1) * self.w/29 + (x+0.5) * self.w * (6/29) - size[0] / 2
            text_y = (y+1) * self.h/29 + (y+0.5) * self.h * (6/29) - size[1] / 2


        return rect, text_render, (text_x, text_y)

    def start(self):

        playing = True
        gameover = False
        bot_active = False
        score  = 0
        moves = 0

        ts = self.env.reset()

        try:
            board_array = ts.observation.numpy()[0]
        except AttributeError:
            board_array = ts.observation
        board_array = np.reshape(board_array, (4,4))

        pygame.display.set_caption(
            "2048" + " " * 10 + "Score: 0   Moves: 0" + self.caption_end
            )
        win = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
        surface = win.copy()

        while playing:

            moved = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    playing = False
                    break

                if event.type == pygame.VIDEORESIZE:
                    self.w, self.h = win.get_size()
                    if self.SQUARE:
                        if self.w > self.h:
                            self.w = self.h
                        else:
                            self.h = self.w
                    surface = win.copy()
                    self.initialize()

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_r:
                        #restart
                        ts = self.env.reset()
                        score = 0
                        moves = 0
                        moved = True
                        gameover = False
                        surface.fill(self.BACKGROUND_COLOR)

                    elif event.key == pygame.K_b:
                        if bot_active:
                            bot_active = False
                            pygame.display.set_caption(
                            "2048" + " " * 10 + f"Score: {int(score)}"
                            + f"   Moves: {moves}"
                            )
                        elif self.bot is not None:
                            bot_active = True
                    elif not gameover:
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

            if not playing:
                break

            if moved:
                try:
                    board_array = ts.observation.numpy()[0]
                    score += ts.reward.numpy()
                except AttributeError:
                    board_array = ts.observation
                    score += ts.reward
                board_array = np.reshape(board_array, (4,4))

            if not gameover:

                if bot_active:
                    old_board = board_array.copy()
                    actionstep = self.bot.action(ts)
                    action = actionstep.action

                    ts = self.env.step(action)
                    moves += 1

                    try:
                        board_array = ts.observation.numpy()[0]
                        score += ts.reward.numpy()
                    except AttributeError:
                        board_array = ts.observation
                        score += ts.reward
                    board_array = np.reshape(board_array, (4,4))

                    if not np.array_equal(old_board, board_array):
                        moved = True
                        time.sleep(self.bot_delay)


                    pygame.display.set_caption(
                        "2048 - Bot" + " " * 10 + f"Score: {int(score)}"
                        + f"   Moves: {moves}"
                        )

                else:
                    pygame.display.set_caption(
                        "2048" + " " * 10 + f"Score: {int(score)}"
                        + f"   Moves: {moves}"+ self.caption_end
                    )


                if ts.is_last():
                    gameover = True

            surface.fill(self.BACKGROUND_COLOR)

            for x in range(4):
                for y in range(4):

                    n = board_array[y][x]
                    rect, text, text_coords = self.tile(x, y, n)

                    try:
                        tile_color = self.TILE_COLOR[n]
                    except KeyError:
                        tile_color = self.TILE_COLOR["BIG"]

                    pygame.draw.rect(
                        surface=surface,
                        color=tile_color,
                        rect=rect,
                        border_radius=int((self.w+self.h)/2 * 1/100)
                        )

                    surface.blit(text, text_coords)

            if gameover:

                x_1 = self.w / 2 - self.gameover_size_1[0] / 2
                y_1 = self.h / 2 - self.h * 6/80
                x_2 = self.w / 2 - self.gameover_size_2[0] / 2
                y_2 = self.h / 2 + self.h * 1/30
                surface.blit(self.gameover_text_1, (x_1, y_1))
                surface.blit(self.gameover_text_2, (x_2, y_2))

            win.fill(self.BACKGROUND_COLOR)

            w, h = win.get_size()
            if w > h:
                win.blit(surface, ((win.get_width()-self.w)/2,0))
            else:
                win.blit(surface, (0, (win.get_height()-self.h)/2))

            pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    from env import PyEnv2048

    game = Game(PyEnv2048)
    game.start()
