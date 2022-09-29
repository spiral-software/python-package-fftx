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


shift = (1,1,1)
target=(0,0,0)

start = (n-shift[0]+target[0],n-shift[1]+target[1],n-shift[2]+target[2])

testSrc = xp.zeros((n,n,n)).astype(src_type)
testSrc[start] = 1.0

symIn = xp.zeros((n*2,n*2,n*2)).astype(src_type)
symIn[shift] = 1.0
testSymHalf = xp.fft.rfftn(symIn)
testSymCube = xp.fft.fftn(symIn)

result = fftx.convo.mdrconv(testSrc, testSymHalf)
print('Using sliced symbol, result[target]   = %.5f' % result[target])

result = fftx.convo.mdrconv(testSrc, testSymCube)
print('Using unsliced symbol, result[target] = %.5f' % result[target])
