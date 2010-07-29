''' This module contains the Population class and Firefly class for
    continuous optimization problems'''
import math as m
import numpy as np
from pylab import *
from BIP.Bayes.lhs import lhs
from scipy.stats import uniform
from multiprocessing import Pool
from functions import *

def write_coords(filename, pop):
    ''' This function will output the points for each fly in the population 
    '''

    with open(filename, 'w') as outputfile:
        for fly in pop.pop:
            for coord in fly.coords:
                outputfile.write(str(coord) + ' ')
            outputfile.write('\n')

class Population:
    ''' The Population class is responsible for maintaining the fireflies and
        running the rounds of optimization
    '''

    NORMAL = 1
    HYBRID = 2
    EPSILON = 1e-5

    _funcs = { 'dejung' : DeJung ,
            'ackley' : Ackley,
            'michalewicz' : Michalewicz,
            'rosenbrock' : Rosenbrock,
            'rastrigin' : Rastrigin}

    def __init__(self, gen, size, alpha, beta, gamma):
        ''' Setup sets the initialization parameters 
            for the population
        '''

        self.gen = gen
        self.size = size
        self.alpha = self.alpha0 = alpha
        self.beta0 = beta
        self.gamma = self.gamma0 = gamma
        self.pop = self.oldpop = None

    def __repr__(self):
        return str(self.pop)

    def __str__(self):
        return str(self.pop)

    def run(self, func_name, dimension_count, style=NORMAL, cpu_count=1, draw=False):
        ''' Run begins the optimization based 
            on the initialization parameters given 
        '''

        #get the actual function
        func = self._funcs[func_name](dimension_count)
        
        #create our populations
        self.pop = self._generate_pop(self.size, func)
        self.oldpop = self._generate_pop(self.size, func)
        
        #scale our gamma 
        self.gamma = self.gamma0 / (func.maxs[0] - func.mins[0])

        #setup drawing if necessary
        if draw:
            ion()
            coords, = plot([fly.coords[0] for fly in self.pop], [fly.coords[1] for fly in self.pop], 'o')
        else:
            coords = None

        if style == Population.NORMAL:
            self._npop(coords, cpu_count)
        else:
            self._hpop(coords, cpu_count)

        self.pop.sort()
        return self.pop[0]

    def test(self, func_name, dimension_count, style=NORMAL, cpu_count=1, draw=False):
        ''' Runs the Firefly algorithm until the mean delta of
            the values is less than EPSILON 
        '''

        #get the actual function
        func = self._funcs[func_name](dimension_count)
        
        #create our populations
        self.pop = self._generate_pop(self.size, func)
        self.oldpop = self._generate_pop(self.size, func)

        #scale our gamma 
        self.gamma = self.gamma0 / (func.maxs[0] - func.mins[0])

        #setup drawing if necessary
        if draw:
            ion()
            coords, = plot([fly.coords[0] for fly in self.pop], [fly.coords[1] for fly in self.pop], 'o')
        else:
            coords = None

        if style == Population.NORMAL:
            count = self._test_population(coords, cpu_count)
            line = 'test did ' + str(count) + ' evaluations'
        else:
            count = self._hybrid_test_population(coords, cpu_count)
            line = 'hybrid test did ' + str(count) + ' evaluations'

        print line
        self.pop.sort()
        return self.pop[0]

    def _generate_pop(self, size, func):
        ''' initializes our population 
        '''
        
        dim = len(func.maxs)
        params = [(func.mins[i], func.maxs[i] - func.mins[i]) for i in xrange(dim)]
        seeds = np.array(lhs([uniform]*dim, params, size, True, np.identity(dim))).T
        seeds.astype(np.float32)

        flies = [FireFly(func, self, dim, np.array(seeds[i])) for i in xrange(size)]

        return flies

    def _npop(self, coords, cpu_count):
        ''' _npop runs the normal firefly algorithm 
        '''

        #initialize our process pool
        pool = Pool(processes=cpu_count)
        for _ in xrange(self.gen):
            #draw if we are supposed to be visualization
            if coords is not None:
                coords.set_xdata([fly.coords[0] for fly in self.pop])
                coords.set_ydata([fly.coords[1] for fly in self.pop])
                draw()

            #copy our population over to old one as well
            self._copy_pop()
            #map our current population to a new one
            self.pop[:] = map(map_fly, self.pop)
            self.pop.sort()


    def _test_population(self, coords, cpu_count):
        ''' runs the optimization until the mean values of change are 
            less than a given epsilon. Returns the amount of function
            evaluations 
        '''

        #initialize our process pool
        i = 0
        pool = Pool(processes=cpu_count)

        while True:
            #draw if we are supposed to be visualization
            if coords is not None:
                coords.set_xdata([fly.coords[0] for fly in self.pop])
                coords.set_ydata([fly.coords[1] for fly in self.pop])
                draw()

            #copy our population over to old one as well
            self._copy_pop()
            
            #map our current population to a new one
            self.pop[:] = pool.map(map_fly, self.pop)

            #calculate the delta of the means
            if self._delta_of_means() < Population.EPSILON:
                break
            
            self.pop.sort()
            i += 1

        return i * len(self.pop) 

    def _hpop(self, coords, cpu_count):
        ''' _hpop runs the hybrid firefly algorithm 
        '''

        #initialize our process pool
        pool = Pool(processes=cpu_count)

        def set_pop(fly):
            '''weird hack to get the pickled population alpha to be set
            '''

            fly.pop = self
            return fly

        #start at 2 for the log function. do same amount of steps
        for i in xrange(2, self.gen + 2):
            #draw if we are supposed to be visualization
            if coords is not None:
                coords.set_xdata([fly.coords[0] for fly in self.pop])
                coords.set_ydata([fly.coords[1] for fly in self.pop])
                draw()
            
            #calculate our new alpha value based on the annealing schedule
            #this may change to allow for a user chosen schedule
            self.alpha = self.alpha0 / m.log(i)
            
            #copy our population over to old one as well
            self._copy_pop()
            
            #annoying hack to get around the cached pickle of pop
            self.pop[:] = [set_pop(fly) for fly in self.pop]

            #map our current population to a new one
            self.pop[:] = map(hybrid_map_fly, self.pop)
            self.pop.sort()

    def _hybrid_test_population(self, coords, cpu_count):
        ''' runs the optimization until the mean values of change are
            less than a given epsilon. Returns the amoung of function
            evaluations 
        '''

        #initialize our process pool
        pool = Pool(processes=cpu_count)
        i = 2.0

        def set_pop(fly):
            '''weird hack to get the pickled population alpha to be set
            '''
            
            fly.pop = self
            return fly

        while True:
            #draw if we are supposed to be visualization
            if coords is not None:
                coords.set_xdata([fly.coords[0] for fly in self.pop])
                coords.set_ydata([fly.coords[1] for fly in self.pop])
                draw()

            #calculate our new alpha value based on the annealing schedule
            #this may change to allow for a user chosen schedule
            self.alpha = self.alpha0 / float(log(i))
            
            #copy our population over to old one as well
            self._copy_pop()

            #annoying hack to get around the cached pickle of pop
            self.pop[:] = [set_pop(fly) for fly in self.pop]

            #map our current population to a new one
            self.pop[:] = pool.map(hybrid_map_fly, self.pop)
            
            #calculate the delta of the means
            if self._delta_of_means() < Population.EPSILON:
                break
            
            self.pop.sort()
            i += 1

        return int(i - 2) * len(self.pop)

    def _copy_pop(self):
        ''' copies the population coords to oldpopulation coords 
        '''

        for fly, oldfly in zip(self.pop, self.oldpop):
            oldfly.copy(fly)

    def _delta_of_means(self):
        ''' calculates the delta of the mean values 
        '''

        average = oldaverage = 0.0
        for fly, oldfly in zip(self.pop, self.oldpop):
            average += fly.val
            oldaverage += oldfly.val

        average = average / len(self.pop)
        oldaverage = oldaverage / len(self.pop)

        return m.fabs(average - oldaverage)

class FireFly:
    ''' A FireFly is a point in hyperdimensional space 
    '''

    BETA_MIN = 0.05

    def __init__(self, objfunc, population, dim, seeds):
        self.func = objfunc
        self.pop = population
        self.coords = seeds
        self.moved = False
        self.val = self.func.eval(self.coords)

    def __cmp__(self, fly):
        if isinstance(fly, FireFly):
            return cmp(self.val, fly.val)
        else:
            return cmp(self.val, fly)

    def __str__(self):
        return ' f(' + str(self.coords) + ') = ' + str(self.val)

    def __repr__(self):
        return ' f(' + str(self.coords) + ') = ' + str(self.val)

    def eval(self):
        ''' eval evaluates the firefly given its current coordinates 
        '''

        self.val = self.func.eval(self.coords)

    def copy(self, fly):
        ''' copy the coordinates and val from fly to this 
        '''

        self.val = fly.val
        self.coords[:] = fly.coords

    def map(self):
        ''' nmap maps a firefly to its new position 
        '''

        #compare ourself to other flies and update
        reduce(flyfold, self.pop.oldpop, self)
        
        #reevaluate ourselves in function space
        self.eval()
        
        return self

    def hybrid_map(self):
        ''' hybrid_map maps a firefly to its new 
            position using the hybrid technique 
        '''
        
        #reset moved to False
        self.moved = False
        
        #compare ourself to other flies and update
        reduce(flyfold, self.pop.oldpop, self)

        #if we didn't move, we are a local best
        #in that case, move a little bit randomly
        if not self.moved:
            self.move_random()

        #reevaluate ourselves in function space
        self.eval()

        return self

    def nfoldf(self, fly):
        ''' our fold function to use over a list of flies 
        '''

        if self.val > fly.val:
            #calculate the distance
            dist = self.calculate_dist(fly)
            #calculate the attractiveness beta
            beta = self.calculate_beta(dist, self.pop.beta0, self.pop.gamma)
            #move towards fly
            self.move(self.pop.alpha, beta, fly)

        return self

    def move(self, alpha, beta, fly):
        ''' moves towards another fly based on the 
            values of alpha and beta 
        '''

        #it would be nice to do another map here
        #but it is kind of hard to do with each coord
        #having its own min/max. I dont like
        #the idea of zipping the list with min/max
        #_then_ mapping
        for i, coord in enumerate(self.coords):
            # calc the temp value to set as coord
            tval = coord + (beta * (fly.coords[i] - coord)) + \
                        (alpha * (uniform.rvs() - 0.5))
            # set as coord if within bounds
            self.coords[i] = self.func.mins[i] if tval < self.func.mins[i] \
                    else self.func.maxs[i] if tval > self.func.maxs[i] else tval
        #we moved
        self.moved = True

    def move_random(self):
        ''' moves a little random bit
        '''

        alpha = self.pop.alpha

        for i, coord in enumerate(self.coords):
            # calc the temp value to set as coord
            tval = coord + (alpha * (uniform.rvs() - 0.5))
            # set as coord if within bounds
            self.coords[i] = self.func.mins[i] if tval < self.func.mins[i] \
                    else self.func.maxs[i] if tval > self.func.maxs[i] else tval

    def calculate_dist(self, fly):
        ''' calculates the euclidean distance between flies 
        '''

        totalsum = 0.0
        for coord, flycoord in zip(self.coords, fly.coords):
            totalsum += (coord - flycoord)**2.0

        return m.sqrt(totalsum)

    def calculate_beta(self, dist, beta0, gamma):
        ''' calculates the value of beta, or attraction 
        '''
        
        beta = (beta0 * m.exp((-gamma) * (dist**2.0)))
        return beta if beta > self.BETA_MIN else self.BETA_MIN

#These functions just help with the higher order functions
def flyfold(fly, otherfly):
    ''' this is the function used for folding over a list of flies 
    '''

    return fly.nfoldf(otherfly)

def map_fly(fly):
    ''' this is a weird workaround to be able to use the pool 
    '''
    
    return fly.map()

def hybrid_map_fly(fly):
    ''' this is a weird workaround to be able to use the pool 
    '''

    return fly.hybrid_map()

