#! python

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import fftx
import sys

FORWARD = True
cxtype = np.cdouble


if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("run-fftn sz [ F|I [ d|s [ GPU|CPU ]]]")
    print("  sz is N or N1,N2,N3")
    print("  F = Forward, I = Inverse")
    print("  d = double, s = single precision")
    sys.exit()

nnn = sys.argv[1].split(',')

n1 = int(nnn[0])
n2 = (lambda:n1, lambda:int(nnn[1]))[len(nnn) > 1]()
n3 = (lambda:n2, lambda:int(nnn[2]))[len(nnn) > 2]()

dims = (n1,n2,n3)
    
if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        FORWARD = False
        
if len(sys.argv) > 3:
    if sys.argv[3] == "s":
        cxtype = np.csingle
        
if len ( sys.argv ) > 4:
    plat_arg = sys.argv[4]
else:
    plat_arg = "GPU" if (cp != None) else "CPU"
    
if plat_arg == "GPU" and (cp != None):
    forGPU = True
else:
    forGPU = False 

# init input
src = np.zeros(dims, cxtype)
for k in range (np.size(src)):
    vr = np.random.random()
    vi = np.random.random()
    src.itemset(k, vr + vi * 1j)

xp = np  
if forGPU:
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

