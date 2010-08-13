''' This module contains the Population class and Firefly class for
    continuous optimization problems'''
import math as m
import numpy as np
from pylab import ion, draw, plot
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
                outputfile.write(str(coord) + ',')
            outputfile.write('\n')

class Population:
    ''' The Population class is responsible for maintaining the fireflies and
        running the rounds of optimization
    '''

    NONE = 1
    BOLTZMANN = 2
    CAUCHY = 3
    FAST = 4
    EPSILON = 1e-5

    _funcs = { 'sphere' : Sphere ,
            'ackley' : Ackley,
            'michalewicz' : Michalewicz,
            'rosenbrock' : Rosenbrock,
            'rastrigin' : Rastrigin,
            'easom' : Easom}

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

    def run(self, func_name, dimension_count, style=NONE, cpu_count=1, draw_graph=False):
        ''' Run begins the optimization based 
            on the initialization parameters given 
        '''

        # prepare before the run
        coords, update = self._prepare_run(func_name, dimension_count, style, draw_graph)

        # run the algorithm
        self._map_pop(coords, update, cpu_count)

        # sort the population and return best
        self.pop.sort()
        return self.pop[0]

    def test(self, func_name, dimension_count, style=NONE, cpu_count=1, draw_graph=False):
        ''' Runs the Firefly algorithm until the mean delta of
            the values is less than EPSILON 
        '''

        # prepare before the run
        coords, update = self._prepare_run(func_name, dimension_count, style, draw_graph)

        # run the algorithm
        i = self._test_map_pop(coords, update, cpu_count)

        print 'ran', i, 'iterations until stopped'

        # sort the population and return the best
        self.pop.sort()
        return self.pop[0]

    def is_within_epsilon(self, fly, func_name, dimension_count, epsilon=0.01):
        ''' this will return |f(x*) - f(x_best)|
            from the latest run
        '''

        # get the objective function
        func = self._funcs[func_name](dimension_count)

        # x* + epsilon value
        xplusep = [x + epsilon for x in func.xstar]
        plusval = func.eval(xplusep)

        # x* - epsilon value
        xminusep = [x - epsilon for x in func.xstar]
        minusval = func.eval(xminusep)

        xstarval = func.eval(func.xstar)

        bestval = fly.val
        plusval = xstarval + plusval
        minusval = xstarval - minusval

        print 'plusval is', plusval, 'minus val is', minusval, 'best is', bestval

        return (bestval <= plusval and bestval >= minusval) or \
                (bestval <= minusval and bestval >= plusval)

    def _generate_pop(self, size, func):
        ''' initializes our population 
        '''
        
        dim = len(func.maxs)
        params = [(func.mins[i], func.maxs[i] - func.mins[i]) for i in xrange(dim)]
        seeds = np.array(lhs([uniform]*dim, params, size, True, np.identity(dim))).T
        seeds.astype(np.float32)

        flies = [FireFly(func, self, dim, np.array(seeds[i])) for i in xrange(size)]

        return flies

    def _get_schedule(self, style):
        ''' this gets our annealing schedule based
            on the style specified
        '''

        if style == Population.NONE:
            update = lambda t: self.alpha0
        elif style == Population.BOLTZMANN:
            update = lambda t: self.alpha0 / m.log(float(t))
        elif style == Population.CAUCHY:
            update = lambda t: self.alpha0 / float(t)
        elif style == Population.FAST:
            # perhaps we should ask for values of quench, m, and n at some point...
            # alpha = alpha0 * exp(-c * t**quench)
            # c = m * exp(-n * quench)
            c = 1.0 * m.exp(-1.0 * 1.0)
            update = lambda t: self.alpha0 * m.exp(-c * (float(t)**(1.0)))
        else:
            raise

        return update


    def _prepare_run(self, func_name, dimension_count, style, draw_graph):
        ''' this prepares the environment before we begin
            a simulation run
        '''

        # get the objective function
        func = self._funcs[func_name](dimension_count)
        
        # create our populations
        self.pop = self._generate_pop(self.size, func)
        self.oldpop = self._generate_pop(self.size, func)
        
        # scale our gamma 
        self.gamma = self.gamma0 / (func.maxs[0] - func.mins[0])

        # setup drawing if necessary
        if draw_graph:
            ion()
            coords, = plot([fly.coords[0] for fly in self.pop], [fly.coords[1] for fly in self.pop], 'o')
        else:
            coords = None

        # create our schedule for alpha
        update = self._get_schedule(style)

        # return coords and schedule function
        return (coords, update)
        
    def _map_pop(self, coords, sched_func, cpu_count):
        ''' _hpop runs the firefly algorithm 
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
            self.alpha = sched_func(i)
            
            #copy our population over to old one as well
            self._copy_pop()
            
            #annoying hack to get around the cached pickle of pop
            self.pop[:] = [set_pop(fly) for fly in self.pop]

            #map our current population to a new one
            self.pop[:] = pool.map(map_fly, self.pop)
            self.pop.sort()

    def _test_map_pop(self, coords, schedule, cpu_count):
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

            # calculate our new alpha value based on the annealing schedule
            # this may change to allow for a user chosen schedule
            self.alpha = schedule(i)
            
            # copy our population over to old one as well
            self._copy_pop()

            # annoying hack to get around the cached pickle of pop
            self.pop[:] = [set_pop(fly) for fly in self.pop]

            # map our current population to a new one
            self.pop[:] = pool.map(map_fly, self.pop)
            
            self.pop.sort()

            # calculate the delta of the means
            if self._delta_of_means() < Population.EPSILON:
                break
            
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

    def map(self, flies):
        ''' map maps a firefly to its new 
            position given a list to compare to
        '''
        
        #reset moved to False
        self.moved = False
        
        #compare ourself to other flies and update
        reduce(flyfold, flies, self)

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

    fly.nfoldf(otherfly)
    return fly

def map_fly(fly):
    ''' this is a weird workaround to be able to use the pool 
    '''

    fly.map(fly.pop.oldpop)
    return fly

