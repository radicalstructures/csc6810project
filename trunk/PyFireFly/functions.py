""" Functions for PyFirefly """

def DeJung(dim):
    return ObjFunc(__dejung, [-5 for i in xrange(dim)], [5 for i in xrange(dim)])

def __dejung(coords):
    powers = [x**2 for x in coords]
    return reduce(lambda x,y: x+y, powers)


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

