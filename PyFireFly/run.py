from firefly import *
from pso import *
from sys import argv
from math import fabs

def main(args):
    f = Population(10, 40, 1.0, 1.0, 1.0)
    pso = PSO(10, 40, 2.0, 2.0)
    for _ in range(int(args[4])):

        # calculate the PSO values
        pso.run(args[1],int(args[2]))
        psoval = pso.delta_of_xstar(args[1], int(args[2]))

        # calculate the FA with no schedule
        f.run(args[1], int(args[2]), Population.NONE, int(args[3]), bool(args[5]))

        nonval = f.delta_of_xstar(args[1], int(args[2]))

        # calculate the FA with the Boltzmann schedule
        f.run(args[1], int(args[2]), Population.BOLTZMANN, int(args[3]), bool(args[5]))
        
        boltzval = f.delta_of_xstar(args[1], int(args[2]))
        
        # calculate the FA with the Cauchy schedule
        f.run(args[1], int(args[2]), Population.CAUCHY, int(args[3]), bool(args[5]))

        cauchval = f.delta_of_xstar(args[1], int(args[2]))

        # calculate the FA with the Fast schedule
        f.run(args[1], int(args[2]), Population.FAST, int(args[3]), bool(args[5]))

        fastval = f.delta_of_xstar(args[1], int(args[2]))

        print psoval, ',', nonval, ',', boltzval, ',', cauchval, ',', fastval

if __name__ == '__main__':
    if len(argv) != 6:
        argv = ['run.py']
        argv.append('dejung')
        argv.append(3)
        argv.append(2)
        argv.append(1)
        argv.append(False)

    argv[5] = argv[5].lower() == 'true'
    main(argv)
