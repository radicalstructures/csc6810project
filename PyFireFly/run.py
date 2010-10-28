from firefly import *
from pso import *
from sys import argv
from math import fabs
import numpy as np

def test(func_name, dimension_count, iteration_count, sample_count):
    ''' this function will run all metaheurstic algorithms against
        the given function for the specified amount of samples.
        the mean value will be output in a file for each algorithm
        for each iteration, showcasing the convergence over time
    '''

    f_n = Population(iteration_count, 40, 0.1, 1.0, 1.0)
    f_b = Population(iteration_count, 40, 0.5, 1.0, 1.0)
    f_c = Population(iteration_count, 40, 1.0, 1.0, 1.0)
    pso = PSO(iteration_count, 40, 2.0, 2.0)
    sma = SA(100, 0.01)
    samples = [[],[],[],[],[]]

    c = lambda a: np.mean(np.array(a), axis=0)

    with open('./data/' + func_name + '.dat', 'w') as file:
        file.write('iters   fa  faboltz facauchy    pso sa\n')
        for _ in range(sample_count):
            samples[0].append(f_n.iter_test(func_name, dimension_count, Population.NONE, 1))
            samples[1].append(f_b.iter_test(func_name, dimension_count, Population.BOLTZMANN, 1))
            samples[2].append(f_c.iter_test(func_name, dimension_count, Population.CAUCHY, 1))
            samples[3].append(pso.iter_test(func_name, dimension_count))
            samples[4].append(sma.iter_test(func_name, dimension_count, iteration_count))

        final = np.array([np.arange(iteration_count)] + [c(data) for data in samples])
        for line in final.T:
            line.tofile(file, sep='\t')
            file.write('\n')

def experiment():
    test('sphere', 32, 40, 2)
    #test('ackley', 128, 2, 100)
    #test('michalewicz', 16, 2, 100)
    #test('rosenbrock', 16, 2, 100)
    #test('rastrigin', 16, 2, 100)
    #test('easom', 2, 2, 100)

if __name__ == '__main__':
    experiment()
