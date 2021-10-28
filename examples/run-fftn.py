#! python

import numpy as np
import fftx

FORWARD = 1
INVERSE = -1

# problem dimensions as a tuple
ns = (16,16,16)

direction = FORWARD

# set to True for CUDA code
options = {'cuda' : False}
    
# init input
src = np.zeros(ns, complex)
for k in range (np.size(src)):
    vr = np.random.random()
    vi = np.random.random()
    src.itemset(k, vr + vi * 1j)
        
if direction == FORWARD:
    resC  = fftx.fft.fftn(src, opts=options)
    resPy = np.fft.fftn(src)
else:
    resC  = fftx.fft.ifftn(src, opts=options)
    resPy = np.fft.ifftn(src)

diffCP = np.max( np.absolute( resPy - resC ) )

print ('Difference between Numpy and FFTX transforms: ' + str(diffCP) )

