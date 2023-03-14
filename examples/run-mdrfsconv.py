#! python
"""
usage: run-mdrfsconv N [ d|s [ GPU|CPU ]]
  N = cube size
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

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print(__doc__.strip())
    sys.exit()

#Cube Size
n = int ( sys.argv[1] )

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

testSrc = xp.random.rand(n,n,n).astype(src_type)
       
symIn = xp.random.rand(n*2,n*2,n*2).astype(src_type)
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
