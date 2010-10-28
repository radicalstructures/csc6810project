''' particle swarm optimization algorithm
    used for comparison against firefly
'''
import numpy as np
from math import fabs
from functions import Function
from BIP.Bayes.lhs import lhs
from scipy.stats import uniform

class PSO(object):
    
    def __init__(self, gen, size, alpha, beta):
        self.gen = int(gen)
        self.size = int(size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.best = None

    def _generate_pop(self, psize, func):
        ''' initializes our population 
        '''
        
        mins = np.array(func.mins)
        maxs = np.array(func.maxs)
        lows = -np.abs(maxs - mins)
        hihs = np.abs(maxs - mins)
        
        # calculate our initial positions
        dim = len(func.maxs)
        params = [(mins[i], maxs[i] - mins[i]) for i in xrange(dim)]
        seeds = np.array(lhs([uniform]*dim, params, psize, True, np.identity(dim))).T
        seeds.astype(np.float32)

        # calculate our initial velocities
        v_seeds = np.array([uniform.rvs(lows[i], hihs[i], size=psize) for i in range(dim)]).T

        return [Particle(func, np.array(seeds[i]), np.array(v_seeds[i])) for i in xrange(psize)]

    def run(self, func_name, dim_count):
        ''' runs the simulation given the function name
            and dimension count
        '''

        func = Function(func_name)(dim_count)
        self.pop = self._generate_pop(self.size, func)
        self.best = min(self.pop)

        for _ in xrange(self.gen):
            self.pop = [p.map_eval(self.best.position, self.alpha, self.beta) for p in self.pop]
            self.best = min(self.pop)

        return self.best

    def iter_test(self, func_name, dim_count):
        ''' Runs the PSO algorithm given the initialization
            parameters, outputting a list of the best values
            from the run
        '''

        bests = []
        func = Function(func_name)(dim_count)
        self.pop  = self._generate_pop(self.size, func)
        p_min     = min(self.pop)
        self.best = np.copy(p_min.position)
        self.best_val = p_min.val

        bests.append(self.best_val)
        for _ in xrange(self.gen):
            self.pop  = [p.map_eval(self.best, self.alpha, self.beta) for p in self.pop]
            p_min     = min(self.pop)

            if p_min.val < self.best_val:
                self.best = np.copy(p_min.position)
                self.best_val = p_min.val

            bests.append(self.best_val)

        return np.array(bests)

class Particle(object):

    def __init__(self, func, seeds, v_seeds):
        self.func = func
        self.position = np.copy(np.array(seeds))
        self.velocity = np.copy(np.array(v_seeds))
        self.best     = np.copy(np.array(seeds))
        self.eval()

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
            self.best = np.copy(self.position)

        return self

    def move(self, global_best, alpha, beta):
        for i, coord in enumerate(self.velocity):
            t_min = self.func.mins[i] * 0.5
            t_max = self.func.maxs[i] * 0.5

            tval = coord + ((alpha*uniform.rvs())*(self.best[i] - self.position[i])) + \
                    ((beta * uniform.rvs()) * (global_best[i] - self.position[i]))
            
            self.velocity[i] = t_min if tval < t_min \
                    else t_max if tval > t_max else tval

        for i, coord in enumerate(self.position):
            tval = coord + self.velocity[i]
            
            self.position[i] = self.func.mins[i] if tval < self.func.mins[i] \
                    else self.func.maxs[i] if tval > self.func.maxs[i] else tval

    
        
