#/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as pp


if len(sys.argv) < 2:
	print("You forgot the file!")
	sys.exit(1)

print(sys.argv[1])
f = h5py.File(sys.argv[1],'r')
dset = f.get("/hist")
a = np.array(dset)

fig = pp.figure()
ax = fig.add_subplot(111)
ax.plot(a,".")
pp.show()

