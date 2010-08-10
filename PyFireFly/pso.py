''' particle swarm optimization algorithm
    used for comparison against firefly
'''
import numpy as np
from math import fabs
from functions import *
from BIP.Bayes.lhs import lhs
from scipy.stats import uniform

class PSO(object):
    
    _funcs = { 'dejung' : DeJung ,
            'ackley' : Ackley,
            'michalewicz' : Michalewicz,
            'rosenbrock' : Rosenbrock,
            'rastrigin' : Rastrigin}

    def __init__(self, gen, size, alpha, beta):
        self.gen = int(gen)
        self.size = int(size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.best = None

    def generate_pop(self, size, func):
        ''' initializes our population 
        '''
        
        dim = len(func.maxs)
        params = [(func.mins[i], func.maxs[i] - func.mins[i]) for i in xrange(dim)]
        seeds = np.array(lhs([uniform]*dim, params, size, True, np.identity(dim))).T
        seeds.astype(np.float32)

        return [Particle(func, np.array(seeds[i])) for i in xrange(size)]

    def run(self, func_name, dim_count):
        ''' runs the simulation given the function name
            and dimension count
        '''

        func = self._funcs[func_name](dim_count)

        self.pop = self.generate_pop(self.size, func)

        for _ in xrange(self.gen):
            self.best = min(self.pop)
            self.pop = [p.map_eval(self.best.position, self.alpha, self.beta) for p in self.pop]

        self.best = min(self.pop)

    def delta_of_xstar(self, func_name, dimension_count):
        ''' this will return |f(x*) - f(x_best)|
        '''

        # get the objective function
        func = self._funcs[func_name](dimension_count)

        # x* value
        val = func.eval(func.xstar)

        # get best fly value
        bestval = min(self.pop).val

        return fabs(val - bestval)

    def _generate_pop(self, size, func):
        ''' initializes our population 
        '''
        
        dim = len(func.maxs)
        params = [(func.mins[i], func.maxs[i] - func.mins[i]) for i in xrange(dim)]
        seeds = np.array(lhs([uniform]*dim, params, size, True, np.identity(dim))).T
        seeds.astype(np.float32)


        
class Particle(object):

    def __init__(self, func, seeds):
        self.func = func
        self.position = np.array(seeds)
        self.velocity = np.array(seeds)
        self.best = np.array(seeds)
        self.val = np.inf
    
    def __cmp__(self, particle):
        if isinstance(particle, Particle):
            return cmp(self.val, particle.val)
        else:
            return cmp(self.val, particle)

    def __str__(self):
        return ' f(' + str(self.position) + ') = ' + str(self.val)

    def __repr__(self):
        return ' f(' + str(self.position) + ') = ' + str(self.val)

    def eval(self):
        self.val = self.func.eval(self.position)

    def map_eval(self, global_best, alpha, beta):
        
        self.move(global_best, alpha, beta)

        tmp = self.val

        self.eval()

        if self.val < tmp:
            self.best = np.array(self.position)

        return self

    def move(self, global_best, alpha, beta):
        for i, coord in enumerate(self.velocity):
            tval = coord + ((alpha*(uniform.rvs()))*(global_best[i] - self.position[i])) + \
                    ((beta * uniform.rvs()) * (self.best[i] - self.position[i]))
            
            self.velocity[i] = self.func.mins[i] if tval < self.func.mins[i] \
                    else self.func.maxs[i] if tval > self.func.maxs[i] else tval

        for i, coord in enumerate(self.position):
            tval = coord + self.velocity[i]
            
            self.position[i] = self.func.mins[i] if tval < self.func.mins[i] \
                    else self.func.maxs[i] if tval > self.func.maxs[i] else tval

        self.eval()

    
        
