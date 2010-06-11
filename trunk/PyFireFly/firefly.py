""" This module contains the Population class and Firefly class for
    continuous optimization problems"""
import random
import math
class Population:
    """ The Population class is responsible for maintaining the fireflies and
        running the rounds of optimization"""

    NORMAL = 1
    HYBRID = 2

    def __init__(self, gen, size, dim, alpha, beta, objfunc, style=NORMAL):

        """ Setup sets the initialization parameters 
            for the population"""

        self.gen = gen
        self.alpha = self.alpha0 = alpha
        self.beta0 = beta
        self.func = objfunc
        self.style = style
        self.rand = random.Random()
        self.rand.seed()
        self.pop = self.__generate_pop(size, dim)

    def run(self):
        """ Run begins the optimization based 
            on the initialization parameters given """
        
        if self.style == Population.NORMAL:
            self.__npop()
        else:
            self.__hpop()

        self.pop.sort()
        return self.pop[0]

    def __generate_pop(self, size, dim):
        """ initializes our population """

        return [FireFly(self.func, [self.rand.random() for x in xrange(dim)]) \
                for i in xrange(size)]

    def __npop(self):
        """ __npop runs the normal firefly algorithm """

        for i in xrange(self.gen):
            self.pop = [self.__nmappop(fly) for fly in self.pop]

    def __hpop(self):
        """ __hpop runs the hybrid firefly algorithm """

        for i in xrange(self.gen):
            self.alpha = self.alpha0 / math.log(i)
            self.pop = [self.__hmappop(fly) for fly in self.pop]

    def __nmappop(self, fly):
        """ ___nmappop maps a firefly to its new position """

        fly = reduce(fly.nfoldf, self.pop, fly)
        fly.eval()
        return fly

    def __hmappop(self, fly):
        """ __hmappop maps a firefly to its new 
            position using the hybrid technique """

        val = fly.val
        fly = reduce(fly.hfoldf, self.pop, fly)
        if val == fly.val:
            fly.move_random(self.alpha, self.rand)
        fly.eval()
        return fly

class FireFly:
    """ A FireFly is a point in hyperdimensional space """

    def __init__(self, objfunc, coords):
        self.coords = coords
        self.func = objfunc
        self.val = self.func.eval(self.coords)

    def eval(self):
        """ eval evaluates the firefly given its current coordinates """

        self.val = self.func.eval(self.coords)

    def move(self, alpha, beta, rand, fly):
        """ moves towards another fly based on the 
            values of alpha and beta """

        for i in xrange(len(self.coords)):
            # calc the temp value to set as coord
            tval = ((1 - beta) * fly.coords[i]) + \
                    (beta * self.coords[i]) + (alpha * (rand.random() - 0.5))
            # set as coord if within bounds
            self.coords[i] = self.func.mins[i] if tval < self.func.mins[i] \
                    else self.func.maxs[i] if tval > self.func.maxs[i] else tval

    def move_random(self, alpha, rand):
        """ moves a little random bit"""

        for i in xrange(len(self.coords)):
            # calc the temp value to set as coord
            tval = self.coords[i] + (alpha * (rand.random() - 0.5))
            # set as coord if within bounds
            self.coords[i] = self.func.mins[i] if tval < self.func.mins[i] \
                    else self.func.maxs[i] if tval > self.func.maxs[i] else tval

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

