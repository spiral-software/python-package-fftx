#! python

import numpy as np
import cupy as cp
import fftx

#Cube Size
N = 81

dims = [N,N,N]
dimsTuple = tuple(dims)


src = np.ones(dimsTuple, dtype=np.double)
for  k in range (np.size(src)):
    src.itemset(k,np.random.random()*10.0)

#convert to CuPy array
src = cp.asarray(src) 

#get amplitudes from src, so results of operation should ~= src
amplitudes = cp.absolute(cp.fft.rfftn(src))

dst = fftx.convo.stepphase(src, amplitudes)

diff = cp.max ( cp.absolute ( src - dst ) )

print ('Diff between src and dst =  ' + str(diff) )
