from firefly import *
from pso import *
from sys import argv
from math import fabs

def run(func_name, dimension_count, cpu_count, iteration_count, draw_graph):
    f = Population(15, 40, 0.5, 1.0, 1.0)
    pso = PSO(15, 40, 2.0, 2.0)
    with open('./data/' + func_name + '.csv', 'w') as file:
        for _ in range(iteration_count):

            # calculate the PSO values
            bestpart = pso.run(func_name, dimension_count)
            psoval = bestpart.val

            # calculate the FA with no schedule
            best = f.run(func_name, dimension_count, Population.NONE, cpu_count, draw_graph)

            nonval = best.val 

            # calculate the FA with the Boltzmann schedule
            best = f.run(func_name, dimension_count, Population.BOLTZMANN, cpu_count, draw_graph)
            
            boltzval = best.val
            
            # calculate the FA with the Cauchy schedule
            best = f.run(func_name, dimension_count, Population.CAUCHY, cpu_count, draw_graph)

            cauchval = best.val

            # calculate the FA with the Fast schedule
            best = f.run(func_name, dimension_count, Population.FAST, cpu_count, draw_graph)

            fastval = best.val

            file.write(str(psoval) + ',' + str(nonval) + ',' + str(boltzval) + ',' + str(cauchval) + ',' + str(fastval))

def experiment():
    run('sphere', 2, 2,100, False)
    run('ackley', 128, 2,100, False)
    run('michalewicz', 16, 2,100, False)
    run('rosenbrock', 16, 2,100, False)
    run('rastrigin', 16, 2,100, False)
    run('easom', 2, 2,100, False)

if __name__ == '__main__':
    experiment()
