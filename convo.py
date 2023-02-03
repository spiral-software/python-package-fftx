"""
FFTX Convo Module
=================

Convolutions and convolution-like functions
"""

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


import snowwhite as sw
from snowwhite.mdrconvsolver import *
from snowwhite.mdrfsconvsolver import *
from snowwhite.stepphasesolver import *

_solver_cache = {}

def mdrconv(src, symbol, dst=None):
    """
    Cyclic convolution
    """
    global _solver_cache
    platform = SW_CPU
    if sw.get_array_module(src) == cp:
        platform = SW_HIP if sw.has_ROCm() else SW_CUDA
    opts = { SW_OPT_PLATFORM : platform }
    N = list(src.shape)[1]
    t = 'd'
    if src.dtype.name == 'float32':
        opts[SW_OPT_REALCTYPE] = 'float'
        t = 's'
    ckey = t + '_mdrconv_' + str(N)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = MdrconvProblem(N)
        solver  = MdrconvSolver(problem, opts)
        _solver_cache[ckey] = solver
    result = solver.solve(src, symbol, dst)
    return result

def mdrfsconv(src, symbol, dst=None):
    """
    Free-space convolution
    """
    global _solver_cache
    platform = SW_CPU
    if sw.get_array_module(src) == cp:
        platform = SW_HIP if sw.has_ROCm() else SW_CUDA
    opts = { SW_OPT_PLATFORM : platform }
    N = list(src.shape)[1]
    t = 'd'
    if src.dtype.name == 'float32':
        opts[SW_OPT_REALCTYPE] = 'float'
        t = 's'
    ckey = t + '_mdrfsconv_' + str(N)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = MdrfsconvProblem(N)
        solver  = MdrfsconvSolver(problem, opts)
        _solver_cache[ckey] = solver
    result = solver.solve(src, symbol, dst)
    return result

def stepphase(src, amplitudes, dst=None):
    global _solver_cache
    platform = SW_CPU
    xp = sw.get_array_module(src)
    if cp == cp:
        platform = SW_HIP if sw.has_ROCm() else SW_CUDA
    opts = { SW_OPT_PLATFORM : platform }
    N = list(src.shape)[1]
    t = 'd'
    if src.dtype.name == 'float32':
        opts[SW_OPT_REALCTYPE] = 'float'
        t = 's'
    ckey = t + '_stepphase_' + str(N)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = StepPhaseProblem(N)
        solver  = StepPhaseSolver(problem, opts)
        _solver_cache[ckey] = solver

    #slice amplitudes if it's a cube   
    shape = amplitudes.shape
    if shape[0] == shape[2]:
        N = shape[0]
        Nx = (N // 2) + 1
        _amps = xp.ascontiguousarray(amplitudes[:, :, :Nx])
    else:
        _amps = amplitudes
        
    return solver.solve(src, _amps, dst)

