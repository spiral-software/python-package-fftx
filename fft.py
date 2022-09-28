
import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import snowwhite as sw
from snowwhite.dftsolver import *
from snowwhite.mddftsolver import *
from snowwhite.mdprdftsolver import *

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
    if sw.get_array_module(src) == cp:
        retkey = retkey + '_CU'
    return retkey
    
def _solver_opts(src):
    platform = SW_CPU
    if sw.get_array_module(src) == cp:
        platform = SW_HIP if sw.has_ROCm() else SW_CUDA
    opts = { SW_OPT_PLATFORM : platform }
    if src.dtype.name in ['complex64', 'float32']:
        opts[SW_OPT_REALCTYPE] = 'float'
    if src.flags.f_contiguous:
        opts[SW_OPT_COLMAJOR] = True
    return opts
    
def fft(src):
    global _solver_cache
    ckey = _solver_key('fft', src)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = DftProblem(src.size, SW_FORWARD)
        solver = DftSolver(problem)
        _solver_cache[ckey] = solver
    result = solver.solve(src)
    return result

def ifft(src):
    global _solver_cache
    ckey = _solver_key('1fft', src)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = DftProblem(src.size, SW_INVERSE)
        solver = DftSolver(problem)
        _solver_cache[ckey] = solver
    result = solver.solve(src)
    return result

def fftn(src):
    global _solver_cache
    try:
        ckey = _solver_key('fftn', src)
    except TypeError as ex:
        print(ex)
        xp = sw.get_array_module(src)
        print('trying ' + xp.__name__ + '.fft.fftn()')
        return xp.fft.fftn(src)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = MddftProblem(list(src.shape), SW_FORWARD)
        solver = MddftSolver(problem, _solver_opts(src))
        _solver_cache[ckey] = solver
    result = solver.solve(src)
    return result

def ifftn(src):
    global _solver_cache
    try:
        ckey = _solver_key('ifftn', src)
    except TypeError as ex:
        print(ex)
        xp = sw.get_array_module(src)
        print('using ' + xp.__name__)
        return xp.fft.ifftn(src)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = MddftProblem(list(src.shape), SW_INVERSE)
        solver = MddftSolver(problem, _solver_opts(src))
        _solver_cache[ckey] = solver
    result = solver.solve(src)
    return result

def rfftn(src):
    global _solver_cache
    try:
        ckey = _solver_key('rfftn', src, ['float64', 'float32'])
    except TypeError as ex:
        print(ex)
        xp = sw.get_array_module(src)
        print('trying ' + xp.__name__ + '.fft.rfftn()')
        return xp.fft.fftn(src)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = MdprdftProblem(list(src.shape), SW_FORWARD)
        solver = MdprdftSolver(problem, _solver_opts(src))
        _solver_cache[ckey] = solver
    result = solver.solve(src)
    return result

def irfftn(src, dims):
    global _solver_cache
    try:
        ckey = _solver_key('irfftn', src)
    except TypeError as ex:
        print(ex)
        xp = sw.get_array_module(src)
        print('trying ' + xp.__name__ + '.fft.irfftn()')
        return xp.fft.fftn(src)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = MdprdftProblem(dims, SW_INVERSE)
        solver = MdprdftSolver(problem, _solver_opts(src))
        _solver_cache[ckey] = solver
    result = solver.solve(src)
    return result
 




