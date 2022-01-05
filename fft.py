
import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import snowwhite as sw
from snowwhite.dftsolver import *
from snowwhite.mddftsolver import *

_solver_cache = {}
_dtype_tags = {'complex128' : 'z', 'complex64' : 'c'}

def _solver_key(tf, src):
    typetag = _dtype_tags.get(src.dtype.name, 0)
    if typetag == 0:
        msg = 'datatype ' + src.dtype.name + ' not supported'
        raise TypeError(msg)
    szstr = 'x'.join([str(n) for n in list(src.shape)])
    retkey = typetag + tf + '_' + szstr
    if sw.get_array_module(src) == cp:
        retkey = retkey + '_CU'
    return retkey
    
def _solver_opts(src):
    opts = {SW_OPT_CUDA : True} if (sw.get_array_module(src) == cp) else {SW_OPT_CUDA : False}
    if src.dtype.name == 'complex64':
        opts[SW_OPT_REALCTYPE] = 'float'
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
        print('using ' + xp.__name__)
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

