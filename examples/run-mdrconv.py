# examples/run-mdrconv.py
#
# Copyright 2018-2023, Carnegie Mellon University
# All rights reserved.
#
# See LICENSE (https://github.com/spiral-software/python-package-fftx/blob/main/LICENSE)

"""
usage: run-mdrconv.py sz [ d|s [ GPU|CPU ]]
    sz is N or N1,N2,N3 all N >= 4, single N implies 3D cube
    d  = double, s = single precision   (default: double precision)
    (GPU is default target unless none exists or no CuPy)                     

Three-dimensional real cyclic convolution
"""

import sys
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import fftx

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

if any(n < 4 for n in dims):
    usage()

src_type = np.double
if len(sys.argv) > 2:
    if sys.argv[2] == "f":
        src_type = np.single

forGPU = (cp != None)
if len ( sys.argv ) > 3:
    if sys.argv[3] == "CPU":
        forGPU = False
        
xp = np
if forGPU:
    xp = cp

testSrc = xp.random.rand(n1,n2,n3).astype(src_type)
       
symIn = xp.random.rand(n1,n2,n3).astype(src_type)
testSymHalf = xp.fft.rfftn(symIn)
testSymCube = xp.fft.fftn(symIn)

#NumPy returns Fortran ordering from FFTs
if not forGPU:
    testSymHalf = np.asanyarray(testSymHalf, order='C')
    testSymCube = np.asanyarray(testSymCube, order='C')

result1 = fftx.convo.mdrconv(testSrc, testSymHalf)
# API converts full cube to half cube
result2 = fftx.convo.mdrconv(testSrc, testSymCube)

diff = xp.max(xp.absolute(result1 - result2))

print('Difference using full- vs. half-cube symbols = ' + str(diff))
