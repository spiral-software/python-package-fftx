#! python
# examples/run-fft.py
#
# Copyright 2018-2023, Carnegie Mellon University
# All rights reserved.
#
# See LICENSE (https://github.com/spiral-software/python-package-fftx/blob/main/LICENSE)


"""
usage: run-fft size [ F|I [ d|s [ GPU|CPU ]]]
  F  = Forward, I = Inverse           (default: Forward)
  d  = double, s = single precision   (default: double precision)
                                    
  (GPU is default target unless none exists or no CuPy)                     
                                    
One-dimensional complex FFT
"""

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import fftx
import sys

FORWARD = True
cxtype = np.cdouble

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print(__doc__.strip())
    sys.exit()

n = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        FORWARD = False

if len(sys.argv) > 3:
    if sys.argv[3] == "s":
        cxtype = np.csingle

if len ( sys.argv ) > 4:
    plat_arg = sys.argv[4]
else:
    plat_arg = "GPU" if (cp != None) else "CPU"
    
if plat_arg == "GPU" and (cp != None):
    forGPU = True
else:
    forGPU = False 

# init input
src = np.zeros(n, cxtype)
for k in range (n):
    vr = np.random.random()
    vi = np.random.random()
    src[k] = vr + vi * 1j

xp = np  
if forGPU:
    src = cp.asarray(src)
    xp = cp
        
if FORWARD:
    resC  = fftx.fft.fft(src)
    resPy = xp.fft.fft(src)
else:
    resC  = fftx.fft.ifft(src)
    resPy = xp.fft.ifft(src)

diffCP = xp.max( np.absolute( resPy - resC ) )

print ('Difference between ' + xp.__name__ + ' and FFTX transforms: ' + str(diffCP) )

