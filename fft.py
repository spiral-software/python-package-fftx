
import numpy as np
from snowwhite.dftsolver import *
from snowwhite.mddftsolver import *

_solver_cache = {}

def _solver_key(tf, src, opts):
    szstr = 'x'.join([str(n) for n in list(src.shape)])
    retkey = tf + '_' + szstr
    if opts.get(SW_OPT_CUDA, False):
        retkey = retkey + '_CU'
    return retkey
    
def fft(src, opts={}):
    global _solver_cache
    ckey = _solver_key('fft', src, opts)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = DftProblem(src.size, SW_FORWARD)
        solver = DftSolver(problem)
        _solver_cache[ckey] = solver
    result = solver.solve(src)
    return result

def ifft(src, opts={}):
    global _solver_cache
    ckey = _solver_key('1fft', src, opts)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = DftProblem(src.size, SW_INVERSE)
        solver = DftSolver(problem)
        _solver_cache[ckey] = solver
    result = solver.solve(src)
    return result

def fftn(src, opts={}):
    global _solver_cache
    ckey = _solver_key('fftn', src, opts)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = MddftProblem(list(src.shape), SW_FORWARD)
        solver = MddftSolver(problem, opts)
        _solver_cache[ckey] = solver
    result = solver.solve(src)
    return result

def ifftn(src, opts={}):
    global _solver_cache
    ckey = _solver_key('ifftn', src, opts)
    solver = _solver_cache.get(ckey, 0)
    if solver == 0:
        problem = MddftProblem(list(src.shape), SW_INVERSE)
        solver = MddftSolver(problem, opts)
        _solver_cache[ckey] = solver
    result = solver.solve(src)
    return result

