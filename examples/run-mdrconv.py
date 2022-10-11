#! python

import sys
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import fftx

#Cube Size
n = 32
if len(sys.argv) > 1:
    n = int ( sys.argv[1] )

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

testSrc = xp.random.rand(n,n,n).astype(src_type)
       
symIn = xp.random.rand(n*2,n*2,n*2).astype(src_type)
testSymHalf = xp.fft.rfftn(symIn)
testSymCube = xp.fft.fftn(symIn)

result1 = fftx.convo.mdrconv(testSrc, testSymHalf)
# API converts full cube to half cube
result2 = fftx.convo.mdrconv(testSrc, testSymCube)

diff = xp.max(xp.absolute(result1 - result2))

print('Difference using full- vs. half-cube symbols = ' + str(diff))
