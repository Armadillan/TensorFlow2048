# Tensorflow2048
 
This was my project for Project Week Fall 2020 at The Putney School.

I'm using tf_agents to build a neural network using DDQN to solve the game 2048.

If you want to see an agent of mine in action, use `visualisation.py`.
It's also a good place to start if you want to look around the code.
I think it's pretty well commented.

`env.py` contains the game logic as a Tensorflow PyEnvironment.\
`agent.py` is for creating and training agents.\
`pg_implementation.py` is a graphical interface for 2048 playable by both humans and robots, implemented in pygame.\
`visualisaton.py` uses `pg_implementation.py` to show agents playing the game.
`data_plots.py` can make nice plots of data collected during training.

You control the pygame interface like this:
Move with arrow keys or WASD.\
Press r to restart.
Press b to turn the bot off and on.

This project depends on the following packages:
- tensorflow
- tf_agents
- numpy
- pygame (only `pg_implementation.py`)
- matplotlib (only `data_plots.py`)
- scipy (only `data_plots.py`)

Feel free to send me any questions at <antonikowalik23@gmail.com>

2048 was originally created by Gabriele Cirulli and can be found [here](https://play2048.co/).\
This software is published under the MIT License, see LICENSE.txt
