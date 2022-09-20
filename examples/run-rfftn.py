#! python

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import fftx

FORWARD = True
DOUBLE  = True

# problem dimensions as a tuple
dims = [32,32,32]

# True/False for CUDA code
genCuda = True
genCuda = genCuda and (cp != None)

if DOUBLE:
    ftype = np.double
    cxtype = np.cdouble
else:
    ftype = np.single
    cxtype = np.csingle
    
if FORWARD:
    src = np.ones(tuple(dims), ftype)
    for  k in range (np.size(src)):
        vr = np.random.random()
        src.itemset(k,vr)
else:
    dims2 = dims.copy()
    z = dims2.pop()
    dims2.append(z // 2 + 1)
    src = np.ones(tuple(dims2), cxtype)
    for  k in range (np.size(src)):
        vr = np.random.random()
        vi = np.random.random()
        src.itemset(k,vr + vi * 1j)

xp = np    
if genCuda:
    src = cp.asarray(src)
    xp = cp
        
if FORWARD:
    resC  = fftx.fft.rfftn(src)
    resPy = xp.fft.rfftn(src)
else:
    resC  = fftx.fft.irfftn(src, dims)
    resPy = xp.fft.irfftn(src, dims)

diffCP = xp.max( xp.absolute( resPy - resC ) )

print ('Difference between ' + xp.__name__ + ' and FFTX transforms: ' + str(diffCP) )

