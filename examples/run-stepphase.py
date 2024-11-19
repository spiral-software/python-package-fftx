#! python
# examples/run-stepphase.py
#
# Copyright 2018-2023, Carnegie Mellon University
# All rights reserved.
#
# See LICENSE (https://github.com/spiral-software/python-package-fftx/blob/main/LICENSE)


"""
usage: run-stepphase  N [ d|s [ GPU|CPU ]]
  N = cube size                       (recommend 81)
  d  = double, s = single precision   (default: double precision)
                                    
  (GPU is default target unless none exists or no CuPy)
"""

import sys
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import fftx

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print(__doc__.strip())
    sys.exit()

#Cube Size
N = int ( sys.argv[1] )

src_type = np.double
if len(sys.argv) > 2:
    if sys.argv[2] == "s":
        src_type = np.single

dims = [N,N,N]
dimsTuple = tuple(dims)

forGPU = (cp != None)

if len ( sys.argv ) > 3:
    if sys.argv[3] == "CPU":
        forGPU = False

xp = np
if forGPU:
    xp = cp       

src = xp.ones(dimsTuple, dtype=src_type)
for i in range(dims[0]):
    for j in range(dims[1]):
        for k in range(dims[2]):
            src[i,j,k] = np.random.random()*10.0

#get amplitudes from src, so results of operation should ~= src

print('Test using half-cube amplitudes')
amplitudes = xp.absolute(xp.fft.rfftn(src))
dst = fftx.convo.stepphase(src, amplitudes)
diff = xp.max(xp.absolute(src - dst))
print('Diff between src and dst =  ' + str(diff))

print('Test using full-cube amplitudes')
amplitudes = xp.absolute(xp.fft.fftn(src))
dst = fftx.convo.stepphase(src, amplitudes)
diff = xp.max(xp.absolute(src - dst))
print('Diff between src and dst =  ' + str(diff))

