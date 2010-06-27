""" Functions for PyFirefly """
import math

def DeJung(dim):
    return ObjFunc(__dejung, [-5 for i in xrange(dim)], [5 for i in xrange(dim)])

def __dejung(coords):
    powers = [x**2 for x in coords]
    return reduce(lambda x,y: x+y, powers)

def Ackley(dim):
    return ObjFunc(__ackley, [-30 for i in xrange(dim)], [30 for i in xrange(dim)])

def __ackley(coords):
    n = len(coords)
    a = 20
    b = 0.2
    c = 2 * math.pi
    s1 = s2 = 0.0

    for i in coords:
        s1 += i**2;
        s2 += math.cos(c*i)

    return -a * math.exp(-b * math.sqrt((1.0/n) * s1)) - \
            math.exp((1.0/n) * s2) + a + math.exp(1)

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

