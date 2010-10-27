''' This module contains the Population class and Firefly class for
    continuous optimization problems'''
import math as m
import numpy as np
from BIP.Bayes.lhs import lhs
from scipy.stats import uniform
from scipy.spatial.distance import euclidean
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

        self.m = 3.0
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

    def run(self, func_name, dimension_count, style=NONE, cpu_count=1):
        ''' Run begins the optimization based 
            on the initialization parameters given 
        '''

        # prepare before the run
        update = self._prepare_run(func_name, dimension_count, style)

        # run the algorithm
        self._map_pop(update, cpu_count)

        # sort the population and return best
        self.pop.sort()
        return self.pop[0]

    def test(self, func_name, dimension_count, style=NONE, cpu_count=1):
        ''' Runs the Firefly algorithm until is_lessthan_eps is true 
            returns the tuple (iterations, is_success)
        '''

        # prepare before the run
        update = self._prepare_run(func_name, dimension_count, style)

        # run the algorithm
        i = self._test_map_pop(update, cpu_count)


        # sort the population and return the best
        self.pop.sort()
        success = is_success_f(self.pop[0].val, self.pop[0].func)
        return (i, success)


    def _generate_pop(self, size, func):
        ''' initializes our population 
        '''
        
        dim = len(func.maxs)
        params = [(func.mins[i], func.maxs[i] - func.mins[i]) for i in xrange(dim)]
        seeds = np.array(lhs([uniform]*dim, params, size, True, np.identity(dim))).T
        seeds.astype(np.float32)

        flies = [FireFly(func, self, np.array(seeds[i])) for i in xrange(size)]

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


    def _prepare_run(self, func_name, dimension_count, style):
        ''' this prepares the environment before we begin
            a simulation run
        '''

        # get the objective function
        func = self._funcs[func_name](dimension_count)
        
        # create our populations
        self.pop = self._generate_pop(self.size, func)
        self.oldpop = self._generate_pop(self.size, func)
        
        # scale our gamma 
        self.gamma = self.gamma0 / ((func.maxs[0] - func.mins[0])**self.m)

        # create our schedule for alpha
        update = self._get_schedule(style)

        # return coords and schedule function
        return update
        
    def _map_pop(self, sched_func, cpu_count):
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
            #calculate our new alpha value based on the annealing schedule
            self.alpha = sched_func(i)
            
            #copy our population over to old one as well
            self._copy_pop()
            
            #annoying hack to get around the cached pickle of pop
            self.pop[:] = [set_pop(fly) for fly in self.pop]

            #map our current population to a new one
            self.pop[:] = pool.map(map_fly, self.pop)
            self.pop.sort()

    def _test_map_pop(self, schedule, cpu_count):
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
            # calculate our new alpha value based on the annealing schedule
            # this may change to allow for a user chosen schedule
            self.alpha = schedule(i)
            
            # copy our population over to old one as well
            self._copy_pop()

            # annoying hack to get around the cached pickle of pop
            #self.pop[:] = [set_pop(fly) for fly in self.pop]

            # map our current population to a new one
            self.pop = [map_fly(fly) for fly in self.pop]
            
            self.pop.sort()

            # calculate the delta of the means
            if self._has_converged(self.pop[0], self.pop[1:]):
                break
            
            i += 1

        return int(i - 2) * len(self.pop)

    def _has_converged(self, fly_best, flies, epsilon=0.01, perc=0.3):
        ''' determines if the population has converged or 
            not, ending a test run
        '''
        conv = True
        for fly in flies[0:int(perc * len(flies))]:
            if not max_dist(fly_best.coords, fly.coords, epsilon):
                conv = False
                break

        return conv

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

    def __init__(self, objfunc, population, seeds):
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
            dist = euclidean(self.coords, fly.coords)
            #calculate the attractiveness beta
            beta = self.calculate_beta(dist, self.pop.beta0, self.pop.gamma, self.pop.m)
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

    def calculate_beta(self, dist, beta0, gamma, degree):
        ''' calculates the value of beta, or attraction 
        '''
        
        beta = (beta0 * m.exp((-gamma) * (dist**degree)))
        return beta if beta > self.BETA_MIN else self.BETA_MIN
        #return beta

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

