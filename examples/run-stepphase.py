#! python

import sys
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import fftx

#Cube Size
N = 81
if len(sys.argv) > 1:
    N = int ( sys.argv[1] )

src_type = np.double
if len(sys.argv) > 2:
    if sys.argv[2] == "f":
        src_type = np.single

dims = [N,N,N]
dimsTuple = tuple(dims)

forGPU = (cp != None)

if len ( sys.argv ) > 3:
    if sys.argv[3] == "CPU":
        forGPU = False
        

#build test input in numpy (cupy does not have itemset)
src = np.ones(dimsTuple, dtype=src_type)
for  k in range (np.size(src)):
    src.itemset(k,np.random.random()*10.0)

xp = np
if forGPU:
    xp = cp
    #convert src to CuPy array
    src = cp.asarray(src) 

#get amplitudes from src, so results of operation should ~= src
amplitudes = xp.absolute(xp.fft.rfftn(src))

dst = fftx.convo.stepphase(src, amplitudes)

diff = xp.max(xp.absolute( src - dst ))

print ('Diff between src and dst =  ' + str(diff) )
