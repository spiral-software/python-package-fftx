#! python
# examples/run-fftn.py
#
# Copyright 2018-2023, Carnegie Mellon University
# All rights reserved.
#
# See LICENSE (https://github.com/spiral-software/python-package-fftx/blob/main/LICENSE)


"""
usage: run-fftn sz [ F|I [ d|s [ GPU|CPU [Fortran]]]
  sz is N or N1,N2,.. all N >= 2, single N implies 3D cube
  F  = Forward, I = Inverse           (default: Forward)
  d  = double, s = single precision   (default: double precision)
  GPU is default target unless none exists or no CuPy
  C ordering is default unless Fortran specified
                                    
  (GPU is default target unless none exists or no CuPy)
  
Multi-dimensional complex FFT
"""

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import fftx
import sys

def usage():
    print(__doc__.strip())
    sys.exit()

# array dimensions
try:
  nnn = sys.argv[1].split(',')
  n1 = int(nnn[0])
  if len(nnn) < 2:
    # default to 3D cube
    dims = [n1,n1,n1]
  else:
    dims = [n1]
    for i in range(1, len(nnn)):
      dims.append(int(nnn[i]))
except:
  usage()
  
if any(n < 2 for n in dims):
    usage()

FORWARD = True    
if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        FORWARD = False

cxtype = np.cdouble        
if len(sys.argv) > 3:
    if sys.argv[3] == "s":
        cxtype = np.csingle
        
if len ( sys.argv ) > 4:
    plat_arg = sys.argv[4]
else:
    plat_arg = "GPU" if (cp != None) else "CPU"
    
order = 'C'
if (len ( sys.argv ) > 5) and (sys.argv[5].lower() == 'fortran'):
    order = 'F'
    
if plat_arg == "GPU" and (cp != None):
    forGPU = True
else:
    forGPU = False 

xp = np  
if forGPU:
    xp = cp

# init input
src = xp.zeros(dims, cxtype, order)
for i in range(dims[0]):
    for j in range(dims[1]):
        for k in range(dims[2]):
            vr = np.random.random()
            vi = np.random.random()
            src[i,j,k] = vr + vi * 1j

        
if FORWARD:
    resC  = fftx.fft.fftn(src)
    resPy = xp.fft.fftn(src)
else:
    resC  = fftx.fft.ifftn(src)
    resPy = xp.fft.ifftn(src)

diffCP = xp.max( xp.absolute( resPy - resC ) )

print ('Difference between ' + xp.__name__ + ' and FFTX transforms: ' + str(diffCP) )

