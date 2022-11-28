#! python

import numpy as np
import fftx
import sys

FORWARD = True
cxtype = np.cdouble

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("run-fftn size [ F|I [ d|s ]]")
    print("  F  = Forward, I = Inverse")
    print("  d  = double, s = single precision")
    sys.exit()

n = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        FORWARD = False

if len(sys.argv) > 3:
    if sys.argv[3] == "s":
        cxtype = np.csingle

# init input
src = np.zeros(n, cxtype)
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

