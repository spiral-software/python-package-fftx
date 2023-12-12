# fftx/convo.py
#
# Copyright 2018-2023, Carnegie Mellon University
# All rights reserved.
#
# See LICENSE (https://github.com/spiral-software/python-package-fftx/blob/main/LICENSE)

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


import spiralpy as sp
from spiralpy.mdrconvsolver import *
from spiralpy.mdrfsconvsolver import *
from spiralpy.stepphasesolver import *

_solver_cache = {}

def mdrconv(src, symbol, dst=None):
    """
    Cyclic convolution
    """
    global _solver_cache
    platform = SP_CPU
    if sp.get_array_module(src) == cp:
        platform = SP_HIP if sp.has_ROCm() else SP_CUDA
    opts = { SP_OPT_PLATFORM : platform }
    ##  N = list(src.shape)[1]
    N = list(src.shape)         ##  [1]
    t = 'd'
    if src.dtype.name == 'float32':
        opts[SP_OPT_REALCTYPE] = 'float'
        t = 's'

    ns = 'x'.join([str(n) for n in N])  ##  problem.dimensions()])
    ##  ckey = t + '_mdrconv_' + str(N)
    ckey = t + '_mdrconv_' + ns
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
    platform = SP_CPU
    if sp.get_array_module(src) == cp:
        platform = SP_HIP if sp.has_ROCm() else SP_CUDA
    opts = { SP_OPT_PLATFORM : platform }
    N = list(src.shape)[1]
    t = 'd'
    if src.dtype.name == 'float32':
        opts[SP_OPT_REALCTYPE] = 'float'
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
    platform = SP_CPU
    xp = sp.get_array_module(src)
    if cp == cp:
        platform = SP_HIP if sp.has_ROCm() else SP_CUDA
    opts = { SP_OPT_PLATFORM : platform }
    N = list(src.shape)[1]
    t = 'd'
    if src.dtype.name == 'float32':
        opts[SP_OPT_REALCTYPE] = 'float'
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

