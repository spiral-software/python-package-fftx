#! python

import sys
from snowwhite.stepphasesolver import *
import numpy as np
import cupy as cp
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

genCuda = True          ##  set as default
genHIP  = False

if len ( sys.argv ) > 3:
    if sys.argv[3] == "CUDA":
        genCuda = True
    elif sys.argv[3] == "HIP":
        genCuda = False
        genHIP = True
    elif sys.argv[3] == "CPU":
        genCuda = False

opts = { SW_OPT_CUDA : genCuda, SW_OPT_HIP : genHIP }

src = np.ones(dimsTuple, dtype=src_type)
for  k in range (np.size(src)):
    src.itemset(k,np.random.random()*10.0)

#convert to CuPy array
src = cp.asarray(src) 

#get amplitudes from src, so results of operation should ~= src
amplitudes = cp.absolute(cp.fft.rfftn(src))

dst = fftx.convo.stepphase(src, amplitudes, opts)

diff = cp.max ( cp.absolute ( src - dst ) )

print ('Diff between src and dst =  ' + str(diff) )
