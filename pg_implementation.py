#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame
import env
import time
import numpy as np

pygame.init()

# Starting dimensions
WIDTH = 600
HEIGHT = 600

tile_font_5 = pygame.font.SysFont("Arial", int(HEIGHT * (5/29)))
tile_font_4 = pygame.font.SysFont("Arial", int(HEIGHT * (4/29)))
tile_font_3 = pygame.font.SysFont("Arial", int(HEIGHT * (3/29)))
tile_font_2 = pygame.font.SysFont("Arial", int(HEIGHT * (2/29)))
tile_font_1 = pygame.font.SysFont("Arial", int(HEIGHT * (1/29)))

BACKGROUND_COLOR = (187, 173, 160)

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


gameover_font_1 = pygame.font.SysFont(
    "Arial", int(HEIGHT * (1/10))
    )
gameover_size_1 = gameover_font_1.size("GAME OVER")
gameover_text_1 = gameover_font_1.render(
    "GAME OVER", True, (20,20,20)
        )
gameover_font_2 = pygame.font.SysFont(
    "Arial", int(HEIGHT * (1/20))
    )
gameover_size_2 = gameover_font_2.size("Press \"r\" to restart")
gameover_text_2 = gameover_font_2.render(
    "Press \"r\" to restart", True, (20,20,20)
        )

def Tile(x, y, n):

    rect_x = (x+1) * WIDTH/29 + x * WIDTH * (6/29)
    rect_y = (y+1) * HEIGHT/29 + y * HEIGHT * (6/29)
    rect = pygame.Rect(
        rect_x, rect_y, WIDTH * (6/29), HEIGHT * (6/29),
        border_radius=60)

    if not n:
        text_render = pygame.Surface((0,0))
        text_x = 0
        text_y = 0
    else:
        text = str(n)
        l = len(text)

        if n in VALUES_WITH_DARK_TEXT:
            text_color = DARK_TEXT_COLOR
        else:
            text_color = LIGHT_TEXT_COLOR

        if l < 3:
            font = tile_font_5
        elif l == 3:
            font = tile_font_4
        elif l == 4:
            font = tile_font_3
        elif l < 7:
            font = tile_font_2
        else:
            font = tile_font_1

        text_render = font.render(text, True, text_color)
        size = font.size(text)

        text_x = (x+1) * WIDTH/29 + (x+0.5) * WIDTH * (6/29) - size[0] / 2
        text_y = (y+1) * HEIGHT/29 + (y+0.5) * HEIGHT * (6/29) - size[1] / 2


    return rect, text_render, (text_x, text_y)

def main(game, bot=None, bot_delay=0.1):

    gameover = False

    if bot is None:
        end = " " * 10 + "NO BOT ATTACHED"
    else:
        end = ""
    ts = game.reset()
    try:
        board_array = ts.observation.numpy()[0]
    except AttributeError:
        board_array = ts.observation
    board_array = np.reshape(board_array, (4,4))

    pygame.display.set_caption("2048" + " " * 10 + "Score: 0   Moves: 0" + end)
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    win.fill(BACKGROUND_COLOR)

    for x in range(4):
        for y in range(4):

            n = board_array[y][x]
            rect, text, text_coords = Tile(x, y, n)

            try:
                tile_color = TILE_COLOR[n]
            except KeyError:
                tile_color = TILE_COLOR["BIG"]

            pygame.draw.rect(
                surface=win,
                color=tile_color,
                rect=rect
                )

            win.blit(text, text_coords)

    pygame.display.flip()

    bot_active = False

    score  = 0

    moves = 0

    playing = True

    while playing:

        if gameover:
            x_1 = WIDTH / 2 - gameover_size_1[0] / 2
            y_1 = HEIGHT / 2 - HEIGHT * 1/10
            x_2 = WIDTH / 2 - gameover_size_2[0] / 2
            y_2 = HEIGHT / 2 + 1/30 * HEIGHT
            win.blit(gameover_text_1, (x_1, y_1))
            win.blit(gameover_text_2, (x_2, y_2))
            pygame.display.flip()

        moved = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                playing = False
                break
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_r:
                        # restart
                        ts = game.reset()
                        score = 0
                        moves = 0
                        moved = True
                        gameover = False
                        win.fill(BACKGROUND_COLOR)

                if bot_active:
                    if event.key == pygame.K_b:
                        bot_active = False
                        pygame.display.set_caption(
                            "2048" + " " * 10 + f"Score: {int(score)}"
                            + f"   Moves: {moves}"
                            )
                else:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        # move up
                        ts = game.step(0)
                        moved = True
                        moves += 1
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        # move right
                        ts = game.step(1)
                        moved = True
                        moves += 1
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        # move down
                        ts = game.step(2)
                        moved = True
                        moves += 1
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        # move left
                        ts = game.step(3)
                        moved = True
                        moves += 1
                    elif event.key == pygame.K_b and bot is not None:
                        bot_active = True

        if moved:
            try:
                board_array = ts.observation.numpy()[0]
                score += ts.reward.numpy()
            except AttributeError:
                board_array = ts.observation
                score += ts.reward
            board_array = np.reshape(board_array, (4,4))

        if not gameover:
            if not playing:
                break

            if bot_active:


                old_board = board_array.copy()
                action = bot.action(ts)

                ts = game.step(action)
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
                    time.sleep(bot_delay)


                pygame.display.set_caption(
                    "2048 - Bot" + " " * 10 + f"Score: {int(score)}"
                    + f"   Moves: {moves}"
                    )

            else:
                pygame.display.set_caption(
                    "2048" + " " * 10 + f"Score: {int(score)}"
                    + f"   Moves: {moves}"+ end
                    )
            if moved:
                win.fill(BACKGROUND_COLOR)
                for x in range(4):
                    for y in range(4):

                        n = board_array[y][x]
                        rect, text, text_coords = Tile(x, y, n)

                        try:
                            tile_color = TILE_COLOR[n]
                        except KeyError:
                            tile_color = TILE_COLOR["BIG"]

                        pygame.draw.rect(
                            surface=win,
                            color=tile_color,
                            rect=rect
                            )

                        win.blit(text, text_coords)

                pygame.display.flip()

                if ts.is_last():
                    print(ts)
                    gameover = True

    pygame.quit()

if __name__ == "__main__":
    main(env.PyEnv2048(0))