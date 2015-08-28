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

ax.plot(x, y, ".")
ax.set_xlabel("$\mathbf{n \cdot dt}$")

print(1 / (T + x))

#~ ax = fig.add_subplot(212)
#~ ax.plot( , y);

plt.show()
