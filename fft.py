
import numpy as np
from snowwhite.dftsolver import *
from snowwhite.mddftsolver import *

def fft(src):
    problem = DftProblem(src.size, SW_FORWARD)
    solver = DftSolver(problem)
    result = solver.solve(src)
    return result

def ifft(src):
    problem = DftProblem(src.size, SW_INVERSE)
    solver = DftSolver(problem)
    result = solver.solve(src)
    return result

def fftn(src, opts={}):
    problem = MddftProblem(list(src.shape), SW_FORWARD)
    solver = MddftSolver(problem, opts)
    result = solver.solve(src)
    return result

def ifftn(src, opts={}):
    problem = MddftProblem(list(src.shape), SW_INVERSE)
    solver = MddftSolver(problem, opts)
    result = solver.solve(src)
    return result

