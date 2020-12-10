#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt

with open("..\\Saved run stats\\Run 9 stats.pkl", "rb") as file:
    data_dict = pickle.load(file)

returns = data_dict["Returns"]
lengths = data_dict["Lengths"]
losses = data_dict["Losses"]

plt.plot(returns)

with open("..\\Saved run stats\\Run 10 stats.pkl", "rb") as file:
    data_dict = pickle.load(file)

returns = data_dict["Returns"]
lengths = data_dict["Lengths"]
losses = data_dict["Losses"]

plt.plot(returns)
