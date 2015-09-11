#/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import process_signal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if len(sys.argv) < 2:
	print("You forgot the file!")
	sys.exit(1)

x,y, T = process_signal.process(sys.argv[1])

fig = plt.figure()

ax = fig.add_subplot(111)

ax.bar(x*1e9, y, .5)
ax.set_xlabel("$\mathbf{n \cdot dt}$")
ax.set_yscale("log")

print(1 / (T + x))

ax = fig.add_subplot(212)
ax.set_yscale("log")
ax.bar( 1/(T + x), y, .5);

plt.show()
