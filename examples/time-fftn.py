
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import time
import fftx

# True/False for CUDA code
genCuda = True
genCuda = genCuda and (cp != None)

FORWARD = 1
INVERSE = -1
# FORWARD/INVERSE transform
direction = FORWARD

# Size of cube
M = 128
ns = (M,M,M)
src = np.zeros(ns, dtype=np.complex128)
for k in range (np.size(src)):
    vr = np.random.random()
    vi = np.random.random()
    src.itemset(k, vr + vi * 1j)

if genCuda:
    xp = cp
    src = cp.asarray(src)
    dev = 'GPU'
    pymod = 'CuPy'
else:
    xp = np
    dev = 'CPU'
    pymod = 'NumPy'

if direction == FORWARD:
    pyfunc = xp.fft.fftn
    fftxfunc = fftx.fft.fftn
    funcname = 'fftn'
else:
    pyfunc = xp.fft.ifftn
    fftxfunc = fftx.fft.ifftn
    funcname = 'ifftn'

    
print('**** Timing ' + funcname + ' on ' + dev + ', data type: ' + src.dtype.name + ', dims: ' + str(ns) + ' ****')
print('')
    

# warmup loop
for i in range(10):
    resPy = pyfunc(src)
# timed loop
print('Timing ' + pymod)
ts = time.perf_counter()
for i in range(10):
    resPy = pyfunc(src)
tf = time.perf_counter()
print(f'{((tf - ts)/10.0):0.6f}')
print('')

pytm = tf-ts

# warmup loop
for i in range(10):
    resC  = fftxfunc(src)
# timed loop
print("Timing FFTX")
ts = time.perf_counter()
for i in range(10):
    resC  = fftxfunc(src)
tf = time.perf_counter()
print(f'{((tf - ts)/10.0):0.6f}')
print('')

fftxtm = tf-ts

diffCP = xp.max( xp.absolute( resPy - resC ) )
print('Difference between ' + pymod + ' and FFTX transforms: ' + str(diffCP))
print('Speedup from ' + pymod + ' to FFTX: ' + f'{(pytm / fftxtm):0.2f}' + 'x')
print('')




