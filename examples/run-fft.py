#! python

import numpy as np
import fftx

FORWARD = 1
INVERSE = -1

# problem size
n = 128

direction = FORWARD
    
# init input
src = np.zeros(n, complex)
for k in range (n):
    vr = np.random.random()
    vi = np.random.random()
    src[k] = vr + vi * 1j
        
if direction == FORWARD:
    resC  = fftx.fft.fft(src)
    resPy = np.fft.fft(src)
else:
    resC  = fftx.fft.ifft(src)
    resPy = np.fft.ifft(src)

diffCP = np.max( np.absolute( resPy - resC ) )

print ('Difference between Numpy and FFTX transforms: ' + str(diffCP) )

