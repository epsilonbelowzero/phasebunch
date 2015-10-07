#/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys

import process_distribution

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if len(sys.argv) < 2:
	print("You forgot the file!")
	sys.exit(1)

dist = process_distribution.process(sys.argv[1])

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(dist[0], dist[1])
#~ ax.plot(data[:i] if not singleCol else data, 'ro')
ax.set_xlabel("$\mathbf{n \cdot dt}$")

plt.show()

