''' simulated annealing algorithm for
    continuous optimization problems
'''

import numpy as np
import math as m
from functions import Function
from BIP.Bayes.lhs import lhs
from scipy.stats import uniform

class SA(object):
    ''' simulated annealing optimizer
    '''

    BOLTZMANN = 1
    CAUCHY = 2
    FAST = 3

    def __init__(self, starting_temp, finish_temp):
        ''' initialize our simulated annealing machine
        '''
        self.t_initial = starting_temp
        self.tfinal = finish_temp

    def run(self, func_name, dim, style=BOLTZMANN):
        ''' initialize our simulated annealing machine
        '''

        objfunc = Function(func_name)(dim)
        schedule = self._get_schedule(style)
        prob = self._get_prob_distr(style)

        state_final = self._anneal(objfunc, schedule, prob)
        return state_final

    def iter_test(self, func_name, dim, iterations, style=BOLTZMANN):
        ''' initialize our simulated annealing machine
        '''

        objfunc = Function(func_name)(dim)
        schedule = self._get_schedule(style)
        prob = self._get_prob_distr(style)

        return self._test_anneal(objfunc, schedule, prob, iterations)

    def _get_state(self, objfunc, state, params):
        ''' setup the initial state
        '''

        def adjust_offset(value, minval, maxval):
            # keep within bounds
            return maxval if value > maxval else minval if value < minval else value

        seeds = np.array([uniform.rvs() - 0.5 for param in params])
        state = np.array([adjust_offset(s + seeds[i], objfunc.mins[i], objfunc.maxs[i]) for i, s in enumerate(state)])
        return state

    def _get_schedule(self, style):
        ''' this gets our annealing schedule based
            on the style specified
        '''

        if style == SA.BOLTZMANN:
            update = lambda k_current: self.t_initial / m.log(float(k_current))
        elif style == SA.CAUCHY:
            update = lambda k_current: self.t_initial / float(k_current)
        elif style == SA.FAST:
            # perhaps we should ask for values of quench, m, and n at some point...
            # alpha = alpha0 * exp(-c * t**quench)
            # c = m * exp(-n * quench)
            c = 1.0 * m.exp(-1.0 * 1.0)
            update = lambda k_current: self.t_initial * m.exp(-c * (float(k_current)**(1.0)))
        else:
            raise

        return update

    def _get_prob_distr(self, style):
        ''' this gets our annealing probability
            distribution based on the style
            specified
        '''

        if style == SA.BOLTZMANN:
            prob_distr = boltz_distr
        elif style == SA.CAUCHY:
            prob_distr = cauchy_distr
        elif style == SA.FAST:
            prob_distr = fast_distr
        else:
            raise

        return prob_distr

    def _anneal(self, objfunc, schedule, probfunc):
        ''' our actual annealing method
        '''

        k_current = 1
        # get our initial state
        dim = len(objfunc.maxs)
        params = [(objfunc.mins[i], func.maxs[i] - func.mins[i]) for i in xrange(dim)]
        seeds = np.array(lhs([uniform]*dim, params, 1, True, np.identity(dim))).T[0]
        state = self._get_state(objfunc, [0.0]*dim, seeds)
        state = [s + objfunc.mins[i] for i, s in enumerate(state)]

        best = np.array(state)
        best_val = objfunc.eval(best)
        t_current = self.t_initial

        while t_current > self.tfinal:
            # run until temperature is low enough
            k_current += 1
            t_current = schedule(k_current)

            state = self._get_state(objfunc, state, [1.0]*dim)
            current_val = objfunc.eval(state)

            if uniform.rvs() < probfunc(t_current, current_val, best_val, dim):
                best = state
                best_val = current_val

        return best

    def _test_anneal(self, objfunc, schedule, probfunc, iterations):
        ''' our actual annealing method
        '''

        bests = []
        k_current = 1
        # get our initial state
        dim = len(objfunc.maxs)
        params = [(objfunc.mins[i], objfunc.maxs[i] - objfunc.mins[i]) for i in xrange(dim)]
        seeds = np.array(lhs([uniform]*dim, params, 1, True, np.identity(dim))).T[0]
        state = self._get_state(objfunc, [0.0]*dim, seeds)
        state = [s + objfunc.mins[i] for i, s in enumerate(state)]

        best = np.array(state)
        start_val = best_val = objfunc.eval(best)
        t_current = self.t_initial

        bests.append(best_val)
        for _ in range(iterations):
            k_current += 1
            t_current = schedule(k_current)
            state = self._get_state(objfunc, state, [1.0]*dim)
            current_val = objfunc.eval(state)

            if uniform.rvs() < probfunc(t_current, current_val, best_val, dim):
                best = state
                best_val = current_val
            bests.append(best_val)

        return np.array(bests)



def boltz_distr(t_current, current_val, best_val, dim):
    probability = 0.0
    if current_val < best_val:
        probability = 1.0
    else:
        probability = m.exp(-((best_val - current_val) / t_current))
    return probability

def cauchy_distr(t_current, current_val, best_val, dim):
    probability = 0.0

    if current_val < best_val:
        probability = 1.0
    else:
        delta = best_val - current_val
        probability = (delta + t_current**2.0)**((dim + 1.0) / 2.0)

    return probability

def fast_distr(t_current, current_val, best_val, dim):
    probability = 0.0

    if current_val < best_val:
        probability = 0.0
    else:
        # this isn't right... but just put it there
        # for now 
        delta = best_val - current_val
        probability = m.exp(delta/t_current)

    return probability

