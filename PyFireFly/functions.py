""" Functions for PyFirefly """
import math

def DeJung(dim):
    """ returns the dejung objective function with 
        mins and maxs defined """
    return ObjFunc(_dejung, [-5.14]*dim, [5.14]*dim)

def Ackley(dim):
    """ returns the ackley objective function with
        mins and maxs defined """
    return ObjFunc(_ackley, [-30.0]*dim, [30.0]*dim)

def Rastrigin(dim):
    """ returns the Rastrigin objective function with
        mins and maxs defined """
    return ObjFunc(_rastrigin, [-5.0]*dim, [5.0]*dim)

def _dejung(coords):
    """ the sphere (dejung) function """
    return sum([x**2 for x in coords])

def _ackley(coords):
    """ the ackley function """
    n = len(coords)
    a = 20
    b = 0.2
    c = 2 * math.pi
    s1 = s2 = 0.0

    for i in coords:
        s1 += i**2
        s2 += math.cos(c*i)

    return -a * math.exp(-b * math.sqrt((1.0/n) * s1)) - \
            math.exp((1.0/n) * s2) + a + math.exp(1)

def _rastrigin(coords):
    """ rastrigin function """
    return 20 + sum([x**2 - 10.0*math.cos(2.0 * math.pi * x) for x in coords])

class ObjFunc:
    """ This is our objective function.
        It contains the function as well as its bounds"""

    def __init__(self, func, mins, maxs):
        self.func = func
        self.mins = mins
        self.maxs = maxs

    def eval(self, coords):
        """ Evaluates the function """
        return self.func(coords)

