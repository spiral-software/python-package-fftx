#! python
# examples/time-fftn.py
#
# Copyright 2018-2023, Carnegie Mellon University
# All rights reserved.
#
# See LICENSE (https://github.com/spiral-software/python-package-fftx/blob/main/LICENSE)


"""
usage: time-fftn sz [ F|I [ d|s [ GPU|CPU ]]]
  sz is N or N1,N2,N3
  F  = Forward, I = Inverse           (default: Forward)
  d  = double, s = single precision   (default: double precision)
                                    
  (GPU is default target unless none exists or no CuPy)
  
Time FFTX vs. NumPy/CuPy three-dimensional complex FFT
"""

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import time
import fftx
import sys

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print(__doc__.strip())
    sys.exit()

nnn = sys.argv[1].split(',')

n1 = int(nnn[0])
n2 = (lambda:n1, lambda:int(nnn[1]))[len(nnn) > 1]()
n3 = (lambda:n2, lambda:int(nnn[2]))[len(nnn) > 2]()

dims = (n1,n2,n3)

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
    
if plat_arg == "GPU" and (cp != None):
    forGPU = True
else:
    forGPU = False 

# init input
src = np.zeros(dims, cxtype)
for k in range (np.size(src)):
    vr = np.random.random()
    vi = np.random.random()
    src.itemset(k, vr + vi * 1j)

xp = np  
if forGPU:
    src = cp.asarray(src)
    xp = cp
    dev = 'GPU'
    pymod = 'CuPy'
else:
    xp = np
    dev = 'CPU'
    pymod = 'NumPy'


if FORWARD:
    pyfunc = xp.fft.fftn
    fftxfunc = fftx.fft.fftn
    funcname = 'fftn'
else:
    pyfunc = xp.fft.ifftn
    fftxfunc = fftx.fft.ifftn
    funcname = 'ifftn'

    
print('**** Timing ' + funcname + ' on ' + dev + ', data type: ' + src.dtype.name + ', dims: ' + str(dims) + ' ****')
print('')
    

# warmup loop
for i in range(10):
    resPy = pyfunc(src)
# timed loop
print('Timing ' + pymod)
ts = time.perf_counter()
for i in range(10):
    resPy = pyfunc(src)
tf = time.perf_counter()
print(f'{((tf - ts)/10.0):0.6f}')
print('')

pytm = tf-ts

#first call will allocate
resC = None
# warmup loop
for i in range(10):
    resC  = fftxfunc(src, resC)
# timed loop
print("Timing FFTX")
ts = time.perf_counter()
for i in range(10):
    resC  = fftxfunc(src, resC)
tf = time.perf_counter()
print(f'{((tf - ts)/10.0):0.6f}')
print('')

fftxtm = tf-ts

diffCP = xp.max( xp.absolute( resPy - resC ) )
print('Difference between ' + pymod + ' and FFTX transforms: ' + str(diffCP))
print('Speedup from ' + pymod + ' to FFTX: ' + f'{(pytm / fftxtm):0.2f}' + 'x')
print('')




