''' Functions for PyFirefly 
'''

import math as m
import pycuda.autoinit
import pycuda.gpuarray as gp
import pycuda.driver as drv
import pycuda.reduction as rk
import numpy as np

def Sphere(dim):
    ''' returns the sphere objective function with 
        mins and maxs defined 
    '''

    return ObjFunc(_sphere, [-5.14]*dim, [5.14]*dim, [0.0]*dim)

def Ackley(dim):
    ''' returns the ackley objective function with
        mins and maxs defined 
    '''

    return ObjFunc(_ackley, [-30.0]*dim, [30.0]*dim, [0.0]*dim)

def Michalewicz(dim):
    ''' returns the michalewicz objective function with
        mins and maxs defined 
    '''

    return ObjFunc(_michalewicz, [0.0]*dim, [m.pi]*dim)

def Rastrigin(dim):
    ''' returns the Rastrigin objective function with
        mins and maxs defined 
    '''

    return ObjFunc(_rastrigin, [-5.0]*dim, [5.0]*dim, [0.0]*dim)

def Rosenbrock(dim):
    ''' returns the Rosenbrock objective function with
        mins and maxs defined
    '''

    return ObjFunc(_rosenbrock, [-5.0]*dim, [10.]*dim, [1.0]*dim)

def Easom(dim):
    ''' returns the Easom objective function with
        mins and maxs defined
    '''
    if dim != 2:
        raise

    return ObjFunc(_easom, [-100.0, -100.0], [100.0, 100.0], [m.pi, m.pi])

def _sphere(coords):
    ''' the sphere (sphere) function 
    '''

    return sum([x**2.0 for x in coords])

def _ackley(coords):
    ''' the ackley function 
    '''

    n = len(coords)
    a = 20.0
    b = 0.2
    c = 2.0 * m.pi
    s1 = s2 = 0.0

    for i in coords:
        s1 += i**2.0
        s2 += m.cos(c*i)

    return -a * m.exp(-b * m.sqrt((1.0/n) * s1)) - \
            m.exp((1.0/n) * s2) + a + m.e

def _rastrigin(coords):
    ''' rastrigin function 
    '''

    return 20 + sum([x**2.0 - 10.0*m.cos(2.0 * m.pi * x) for x in coords])

def _rosenbrock(coords):
    ''' rosenbrock function
    '''

    return sum([100.0 * (coords[j]**2.0 - coords[j+1])**2.0 + (coords[j] - 1.0)**2.0 \
            for j in xrange(len(coords) - 1)])

def _michalewicz(coords):
    ''' michalewicz function
    '''
    m = 10.0
    return -sum([m.sin(coord) * (m.sin(i * coord**2.0 / m.pi))**(2.0 * m) \
            for i, coord in enumerate(coords)])

def _easom(coords):
    ''' easom function
    '''
    x1 = coords[0]
    x2 = coords[1]
    return -m.cos(x1) * m.cos(x2) * m.exp((-(x1 - m.pi)**2.0) - ((x2 - m.pi)**2.0))

class ObjFunc:
    ''' This is our objective function.
        It contains the function as well as its bounds
    '''

    def __init__(self, func, mins, maxs, xstar):
        self.func = func
        self.mins = mins
        self.maxs = maxs
        self.xstar = xstar

    def eval(self, coords):
        ''' Evaluates the function 
        '''

        return self.func(coords)

