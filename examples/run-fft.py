#! python

import numpy as np
import fftx

FORWARD = True

# problem size
n = 128

# init input
src = np.zeros(n, complex)
for k in range (n):
    vr = np.random.random()
    vi = np.random.random()
    src[k] = vr + vi * 1j
        
if FORWARD:
    resC  = fftx.fft.fft(src)
    resPy = np.fft.fft(src)
else:
    resC  = fftx.fft.ifft(src)
    resPy = np.fft.ifft(src)

diffCP = np.max( np.absolute( resPy - resC ) )

print ('Difference between Numpy and FFTX transforms: ' + str(diffCP) )

