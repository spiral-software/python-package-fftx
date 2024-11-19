#! python
# examples/run-rfftn.py
#
# Copyright 2018-2023, Carnegie Mellon University
# All rights reserved.
#
# See LICENSE (https://github.com/spiral-software/python-package-fftx/blob/main/LICENSE)


"""
usage: run-rfftn sz [ F|I [ d|s [ GPU|CPU ]]]
  sz is N or N1,N2,N3                 (dimensions of real array)
  F  = Forward, I = Inverse           (default: Forward)
  d  = double, s = single precision   (default: double precision)
                                    
  (GPU is default target unless none exists or no CuPy)
  
Three-dimensional real-to-complex FFT (inverse is complex-to-real)
"""

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import fftx
import sys

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print(__doc__.strip())
    sys.exit()

nnn = sys.argv[1].split(',')

n1 = int(nnn[0])
n2 = (lambda:n1, lambda:int(nnn[1]))[len(nnn) > 1]()
n3 = (lambda:n2, lambda:int(nnn[2]))[len(nnn) > 2]()

dims = [n1,n2,n3]

FORWARD = True    
if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        FORWARD = False

ftype = np.double
cxtype = np.cdouble       
if len(sys.argv) > 3:
    if sys.argv[3] == "s":
        ftype = np.single
        cxtype = np.csingle
        
if len ( sys.argv ) > 4:
    plat_arg = sys.argv[4]
else:
    plat_arg = "GPU" if (cp != None) else "CPU"
    
if plat_arg == "GPU" and (cp != None):
    forGPU = True
else:
    forGPU = False 

if FORWARD:
    # real to complex
    src = np.ones(tuple(dims), ftype)
    for  k in range (np.size(src)):
        vr = np.random.random()
        src.itemset(k,vr)
else:
    # complex to real
    dims2 = dims.copy()
    z = dims2.pop()
    dims2.append(z // 2 + 1)
    src = np.ones(tuple(dims2), cxtype)
    for  k in range (np.size(src)):
        vr = np.random.random()
        vi = np.random.random()
        src.itemset(k,vr + vi * 1j)

xp = np    
if forGPU:
    src = cp.asarray(src)
    xp = cp
        
if FORWARD:
    resC  = fftx.fft.rfftn(src)
    resPy = xp.fft.rfftn(src)
else:
    resC  = fftx.fft.irfftn(src, dims)
    resPy = xp.fft.irfftn(src, dims)

diffCP = xp.max( xp.absolute( resPy - resC ) )

print ('Difference between ' + xp.__name__ + ' and FFTX transforms: ' + str(diffCP) )

