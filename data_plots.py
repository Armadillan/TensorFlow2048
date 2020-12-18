#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This script was modified to produce a variety of plots.
Its current state makes only some of them.

It can be edited if one wishes to create different plots.

"""

import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np


with open("..\\Saved run stats\\run 8 stats.pkl", "rb") as file:
    data_dict = pickle.load(file)

returns9 = data_dict["Returns"]
lengths9 = data_dict["Lengths"]
losses9 = data_dict["Losses"]


# with open("..\\Saved run stats\\Run 10 stats.pkl", "rb") as file:
#     data_dict = pickle.load(file)

# returns10 = data_dict["Returns"]
# lengths10 = data_dict["Lengths"]
# losses10 = data_dict["Losses"]

x = np.linspace(0, 10, len(returns9))

fig, returns_plot = plt.subplots()

returns_plot.plot(x, returns9)
returns9 = gaussian_filter1d(returns9, 6)
returns_plot.plot(x, returns9)

returns_plot.set_title("Run 8 average returns")
returns_plot.set_xlabel("Millions of iterations")
returns_plot.set_ylabel("Returns")
returns_plot.legend(["Original data", "Filtered data"], loc="best")
returns_plot.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))


fig, lengths_plot = plt.subplots()

lengths_plot.plot(x, lengths9)
lengths9 = gaussian_filter1d(lengths9, 6)
lengths_plot.plot(x, lengths9)

lengths_plot.set_title("Run 8 average episode lengths")
lengths_plot.set_xlabel("Millions of iterations")
lengths_plot.set_ylabel("Iterations per episode")
lengths_plot.legend(["Original data", "Filtered data"], loc="best")
lengths_plot.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))


fig, losses_plot = plt.subplots()

x_losses = np.linspace(0, 10, len(losses9))

losses_plot.plot(x_losses, losses9)


losses_plot.set_title("Run 8 loss")
losses_plot.set_xlabel("Millions of iterations")
losses_plot.set_ylabel("Loss")
losses_plot.xaxis.set_ticks(np.arange(min(x_losses), max(x_losses)+1, 1.0))


# returns9 = gaussian_filter1d(returns9, 6)
# returns10 = gaussian_filter1d(returns10, 6)

# lengths9 = gaussian_filter1d(lengths9, 6)
# lengths10 = gaussian_filter1d(lengths10, 6)

# fig, axes = plt.subplots(2,2, sharey="row", sharex="col", figsize=(9.6, 7.2))

# returns9ax = axes[0][0]
# returns10ax = axes[0][1]
# lengths9ax = axes[1][0]
# lengths10ax = axes[1][1]

# returns9ax.plot(x, returns9)
# returns10ax.plot(x, returns10)
# lengths9ax.plot(x, lengths9)
# lengths10ax.plot(x, lengths10)

# returns9ax.set_title("Run 9 returns")
# returns9ax.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))
# returns9ax.set_ylabel("Returns")

# returns10ax.set_title("Run 10 returns")
# returns10ax.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))

# lengths9ax.set_title("Run 9 episode lengths")
# lengths9ax.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))
# lengths9ax.set_ylabel("Iterations per episode")
# lengths9ax.set_xlabel("Millions of iterations")

# lengths10ax.set_title("Run 10 episode lengths")
# lengths10ax.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))
# lengths10ax.set_xlabel("Millions of iterations")


# # lengths9 = gaussian_filter1d(lengths9, 6)
# # lengths10 = gaussian_filter1d(lengths10, 6)
# fig, lengths_plot = plt.subplots()

# lengths_plot.plot(x, lengths9)
# lengths_plot.plot(x, lengths10)

# lengths_plot.set_title("Average episode lengths (filtered data)")
# lengths_plot.set_xlabel("Millions of iterations")
# lengths_plot.set_ylabel("Iterations per episode")
# lengths_plot.legend(["Run 9", "Run 10"], loc="lower left")
# lengths_plot.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))

# lengths_plot.set_xlim(4.5, 10.5)
