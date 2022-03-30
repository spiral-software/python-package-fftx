"""
FFTX Convo Module
=================

Convolutions and convolution-like functions

Requires CuPy
"""

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


import snowwhite as sw
from snowwhite.stepphasesolver import *

_solver_cache = {}

def stepphase(src, amplitudes):
    global _solver_cache
    N = list(src.shape)[1]
    t = 'd'
    opts = {SW_OPT_CUDA : True}
    if src.dtype.name == 'float32':
        opts[SW_OPT_REALCTYPE] = 'float'
        t = 's'
    ckey = t + '_stepphase_' + str(N)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = StepPhaseProblem(N)
        solver  = StepPhaseSolver(problem, opts)
        _solver_cache[ckey] = solver
    result = solver.solve(src, amplitudes)
    return result

