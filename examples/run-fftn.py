#! python

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import fftx

FORWARD = True

# problem dimensions as a tuple
ns = (16,16,16)

# True/False for CUDA code
genCuda = True
genCuda = genCuda and (cp != None)
    
# init input
src = np.zeros(ns, complex)
for k in range (np.size(src)):
    vr = np.random.random()
    vi = np.random.random()
    src.itemset(k, vr + vi * 1j)

xp = np    
if genCuda:
    src = cp.asarray(src)
    xp = cp
        
if FORWARD:
    resC  = fftx.fft.fftn(src)
    resPy = xp.fft.fftn(src)
else:
    resC  = fftx.fft.ifftn(src)
    resPy = xp.fft.ifftn(src)

diffCP = xp.max( xp.absolute( resPy - resC ) )

print ('Difference between ' + xp.__name__ + ' and FFTX transforms: ' + str(diffCP) )

