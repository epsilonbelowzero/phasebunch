#/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import h5py
from mpl_toolkits.mplot3d import Axes3D

if len(sys.argv) < 2:
	print("You forgot the file!")
	sys.exit(1)

print(sys.argv[1])
f = h5py.File(sys.argv[1],'r')
dset = f["/hist"]

print(dset[0])
print(len(dset))

