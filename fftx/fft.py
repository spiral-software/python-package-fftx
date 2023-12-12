# fftx/fft.py
#
# Copyright 2018-2023, Carnegie Mellon University
# All rights reserved.
#
# See LICENSE (https://github.com/spiral-software/python-package-fftx/blob/main/LICENSE)


import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import spiralpy as sp
from spiralpy.dftsolver import *
from spiralpy.mddftsolver import *
from spiralpy.mdprdftsolver import *

_solver_cache = {}
_dtype_tags = {'complex128' : 'z', 'complex64' : 'c', 'float64' : 'd', 'float32' : 's'}

def _solver_key(tf, src, types=['complex128', 'complex64']):
    if src.dtype.name in types:
        typetag = _dtype_tags.get(src.dtype.name, 0)
    else:
        typetag = 0
    if typetag == 0:
        msg = 'datatype ' + src.dtype.name + ' not supported by fftx.' + tf + '()'
        raise TypeError(msg)
    szstr = 'x'.join([str(n) for n in list(src.shape)])
    retkey = typetag + tf + '_' + szstr
    if src.flags.f_contiguous:
        retkey = retkey + 'F'
    if sp.get_array_module(src) == cp:
        retkey = retkey + '_CU'
    return retkey
    
def _solver_opts(src):
    platform = SP_CPU
    if sp.get_array_module(src) == cp:
        platform = SP_HIP if sp.has_ROCm() else SP_CUDA
    opts = { SP_OPT_PLATFORM : platform }
    if src.dtype.name in ['complex64', 'float32']:
        opts[SP_OPT_REALCTYPE] = 'float'
    if src.flags.f_contiguous:
        opts[SP_OPT_COLMAJOR] = True
    return opts
    
def fft(src, dst=None):
    global _solver_cache
    ckey = _solver_key('fft', src)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = DftProblem(src.size, SP_FORWARD)
        solver = DftSolver(problem, _solver_opts(src))
        _solver_cache[ckey] = solver
    result = solver.solve(src, dst)
    return result

def ifft(src, dst=None):
    global _solver_cache
    ckey = _solver_key('1fft', src)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = DftProblem(src.size, SP_INVERSE)
        solver = DftSolver(problem, _solver_opts(src))
        _solver_cache[ckey] = solver
    result = solver.solve(src, dst)
    return result

def fftn(src, dst=None):
    global _solver_cache
    
    #for GPU use CuPy function if number of dims is not 3
    if (len(src.shape) != 3) and (sp.get_array_module(src) == cp):
        return cp.fft.fftn(src)

    try:
        ckey = _solver_key('fftn', src)
    except TypeError as ex:
        print(ex)
        xp = sp.get_array_module(src)
        print('trying ' + xp.__name__ + '.fft.fftn()')
        return xp.fft.fftn(src)
    solver = _solver_cache.get(ckey, 0)
    try:
        if solver == 0:
            problem = MddftProblem(list(src.shape), SP_FORWARD)
            solver = MddftSolver(problem, _solver_opts(src))
            _solver_cache[ckey] = solver
        return solver.solve(src, dst)
    except:
        xp = sp.get_array_module(src)
        print('trying ' + xp.__name__ + '.fft.fftn()')
        return xp.fft.fftn(src)

def ifftn(src, dst=None):
    global _solver_cache
    
    #for GPU use CuPy function if number of dims is not 3
    if (len(src.shape) != 3) and (sp.get_array_module(src) == cp):
        return cp.fft.ifftn(src)

    try:
        ckey = _solver_key('ifftn', src)
    except TypeError as ex:
        print(ex)
        xp = sp.get_array_module(src)
        print('using ' + xp.__name__)
        return xp.fft.ifftn(src)
    solver = _solver_cache.get(ckey, 0)
    try:
        if solver == 0:
            problem = MddftProblem(list(src.shape), SP_INVERSE)
            solver = MddftSolver(problem, _solver_opts(src))
            _solver_cache[ckey] = solver
        result = solver.solve(src, dst)
    except:
        xp = sp.get_array_module(src)
        print('using ' + xp.__name__)
        return xp.fft.ifftn(src)
    return result

def rfftn(src, dst=None):
    global _solver_cache
    try:
        ckey = _solver_key('rfftn', src, ['float64', 'float32'])
    except TypeError as ex:
        print(ex)
        xp = sp.get_array_module(src)
        print('trying ' + xp.__name__ + '.fft.rfftn()')
        return xp.fft.fftn(src)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = MdprdftProblem(list(src.shape), SP_FORWARD)
        solver = MdprdftSolver(problem, _solver_opts(src))
        _solver_cache[ckey] = solver
    result = solver.solve(src, dst)
    return result

def irfftn(src, dims, dst=None):
    global _solver_cache
    try:
        ckey = _solver_key('irfftn', src)
    except TypeError as ex:
        print(ex)
        xp = sp.get_array_module(src)
        print('trying ' + xp.__name__ + '.fft.irfftn()')
        return xp.fft.fftn(src)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = MdprdftProblem(dims, SP_INVERSE)
        solver = MdprdftSolver(problem, _solver_opts(src))
        _solver_cache[ckey] = solver
    result = solver.solve(src, dst)
    return result
 




