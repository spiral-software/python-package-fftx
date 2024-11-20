#! python
# examples/run-mdrfsconv.py
#
# Copyright 2018-2023, Carnegie Mellon University
# All rights reserved.
#
# See LICENSE (https://github.com/spiral-software/python-package-fftx/blob/main/LICENSE)

"""
usage: run-mdrfsconv.py sz [ d|s [ GPU|CPU ]]
    sz is N or N1,N2,N3 all N >= 4, single N implies 3D cube
    d  = double, s = single precision   (default: double precision)
    (GPU is default target unless none exists or no CuPy)       
                                    
Three-dimensional real free-space convolution
"""

import sys
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

##  array dimensions
try:
    nnn = sys.argv[1].split(',')
    n1 = int(nnn[0])
    n2 = (lambda:n1, lambda:int(nnn[1]))[len(nnn) > 1]()
    n3 = (lambda:n2, lambda:int(nnn[2]))[len(nnn) > 2]()
    dims = [n1,n2,n3]
except:
    usage()

src_type = np.double
cplx_type = np.complex128
if len(sys.argv) > 2:
    if sys.argv[2] == "s":
        src_type = np.single
        cplx_type = np.complex64

forGPU = (cp != None)
if len ( sys.argv ) > 3:
    if sys.argv[3] == "CPU":
        forGPU = False
        
xp = np
if forGPU:
    xp = cp

testSrc = xp.random.rand(n1,n2,n3).astype(src_type)
       
symIn = xp.random.rand(n1*2,n2*2,n3*2).astype(src_type)
testSymHalf = xp.fft.rfftn(symIn)
testSymCube = xp.fft.fftn(symIn)

#NumPy returns Fortran ordering from FFTs and always complex128
if not forGPU:
    testSymHalf = np.asanyarray(testSymHalf, dtype=cplx_type, order='C')
    testSymCube = np.asanyarray(testSymCube, dtype=cplx_type, order='C')

result1 = fftx.convo.mdrfsconv(testSrc, testSymHalf)
# API converts full cube to half cube
result2 = fftx.convo.mdrfsconv(testSrc, testSymCube)

diff = xp.max(xp.absolute(result1 - result2))

print('Difference using full- vs. half-cube symbols = ' + str(diff))
